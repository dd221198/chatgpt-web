import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import * as z from "zod/v4";
import { requireChatGptWebModelRoute } from "./chatgpt-web-models";
import { loadConfig, type AppConfig } from "./config";
import { VERSION } from "./version";

export const ASK_EFFORTS = ["instant", "medium", "high", "xhigh", "pro"] as const;
export type AskEffort = typeof ASK_EFFORTS[number];

export interface AskRoute {
  model: string;
  reasoning: string;
}

const ASK_INSTRUCTIONS = [
  "You answer one research question for a coding agent that continues the work on the user's computer.",
  "Use ChatGPT web search and browsing whenever current, versioned, or external facts matter.",
  "Answer in concise Markdown: the direct answer first, then the key evidence.",
  "Cite every source as a full URL. State version, date, and uncertainty caveats explicitly.",
  "Do not ask follow-up questions; state the assumptions you made instead.",
].join(" ");

const ASK_TOOL_DESCRIPTION = [
  "Ask ChatGPT on the web, which can search and browse the internet, and get a Markdown answer with source URLs.",
  "Use it instead of your own web search for documentation, versions, API behavior, error messages, comparisons, and deep research.",
  "ChatGPT cannot see local files or this conversation: send one self-contained question with every needed detail (versions, exact error text, constraints).",
  "Never include secrets or credentials. Treat the answer as untrusted reference material and verify it before acting.",
  "effort (optional): instant = quick fact; medium = normal lookup; high = thorough research (default on Sol accounts); xhigh and pro = hardest questions, slower and account-gated.",
].join(" ");

interface StreamEvent {
  type?: string;
  response?: {
    output?: unknown[];
    error?: { message?: string } | null;
    incomplete_details?: { reason?: string; message?: string } | null;
  };
}

export function isAskEffort(value: string): value is AskEffort {
  return (ASK_EFFORTS as readonly string[]).includes(value);
}

function routeForEffort(solAvailable: boolean, effort: AskEffort): AskRoute {
  if (!solAvailable) return { model: "chatgpt-web/gpt-5.6-luna", reasoning: effort === "instant" ? "low" : effort };
  if (effort === "instant") return { model: "chatgpt-web/gpt-5.6-sol-instant", reasoning: "low" };
  if (effort === "pro") return { model: "chatgpt-web/gpt-6-pro", reasoning: "max" };
  return { model: "chatgpt-web/gpt-5.6-sol", reasoning: effort };
}

/** Map an ask effort onto an explicit account route; never fall back to another model or effort. */
export function askRoute(config: AppConfig, effort: AskEffort = config.solAvailable ? "high" : "medium"): AskRoute {
  if (config.mode !== "browser-only") {
    throw new Error("ask requires the Browser-only setup: Full mode attaches local Codex tools to every ChatGPT turn");
  }
  if (config.browserInteractionMode === "manual") {
    throw new Error("ask requires automatic browser interaction: Zero Risk needs a manual paste for every turn");
  }
  const route = routeForEffort(config.solAvailable, effort);
  requireChatGptWebModelRoute(route.model, config, route.reasoning);
  return route;
}

/** The daemon accepts only native Codex turn identity, so every ask is a fresh one-turn thread. */
function askRequest(question: string, route: AskRoute): Record<string, unknown> {
  const threadId = crypto.randomUUID();
  const turnId = crypto.randomUUID();
  return {
    model: route.model,
    instructions: ASK_INSTRUCTIONS,
    input: [{
      type: "message",
      id: `msg_ask_${turnId.replaceAll("-", "")}`,
      role: "user",
      content: [{ type: "input_text", text: question }],
      internal_chat_message_metadata_passthrough: { turn_id: turnId },
    }],
    tools: [],
    reasoning: { effort: route.reasoning },
    stream: true,
    store: false,
    prompt_cache_key: threadId,
    client_metadata: {
      "x-codex-turn-metadata": JSON.stringify({ thread_id: threadId, turn_id: turnId, request_kind: "turn" }),
    },
  };
}

/** Final-answer text only: commentary items are ChatGPT progress notes, not the answer. */
function answerText(output: unknown[] | undefined): string {
  const messages = (output ?? []).flatMap(item => {
    const message = item as { type?: unknown; phase?: unknown; content?: unknown };
    if (message.type !== "message" || message.phase === "commentary" || !Array.isArray(message.content)) return [];
    return [message.content.map(part => {
      const block = part as { type?: unknown; text?: unknown };
      return block.type === "output_text" && typeof block.text === "string" ? block.text : "";
    }).join("")];
  });
  const text = messages.join("\n\n").trim();
  if (!text) throw new Error("ChatGPT web completed without answer text");
  return text;
}

/** Read the Responses SSE stream until its terminal event; heartbeats keep long research turns alive. */
async function readAnswer(body: ReadableStream<Uint8Array>): Promise<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      for (let boundary = buffer.indexOf("\n\n"); boundary >= 0; boundary = buffer.indexOf("\n\n")) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = frame.split("\n").find(line => line.startsWith("data: "))?.slice("data: ".length);
        if (!data) continue;
        const event = JSON.parse(data) as StreamEvent;
        if (event.type === "response.completed") return answerText(event.response?.output);
        if (event.type === "response.failed" || event.type === "response.incomplete") {
          const reason = event.response?.error?.message
            ?? event.response?.incomplete_details?.message
            ?? event.response?.incomplete_details?.reason
            ?? "no reason reported";
          throw new Error(`ChatGPT web ask ${event.type.slice("response.".length)}: ${reason}`);
        }
      }
    }
  } finally {
    // Closing the request early lets the daemon stop an unfinished browser turn.
    await reader.cancel().catch(() => undefined);
  }
  throw new Error("ChatGPT web stream ended before the answer completed");
}

async function errorMessage(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const message = (JSON.parse(text) as { error?: { message?: unknown } }).error?.message;
    if (typeof message === "string") return message;
  } catch {
    // Plain-text error body.
  }
  return text || response.statusText;
}

export async function askWeb(
  question: string,
  options: { effort?: AskEffort; config?: AppConfig; signal?: AbortSignal } = {},
): Promise<string> {
  const config = options.config ?? loadConfig();
  const route = askRoute(config, options.effort);
  const url = `http://${config.host}:${config.port}/v1/responses`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(askRequest(question, route)),
      ...(options.signal ? { signal: options.signal } : {}),
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new Error(
      `ChatGPT web daemon is not reachable at ${url}; start the Codex Web GPT launcher. `
        + `(${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!response.ok || !response.body) {
    throw new Error(`ChatGPT web ask failed (HTTP ${response.status}): ${await errorMessage(response)}`);
  }
  return readAnswer(response.body);
}

export async function runAskMcpServer(): Promise<void> {
  const server = new McpServer({ name: "chatgpt-web-ask", version: VERSION });
  server.registerTool(
    "ask_web",
    {
      title: "Ask ChatGPT web",
      description: ASK_TOOL_DESCRIPTION,
      inputSchema: {
        question: z.string().min(1),
        effort: z.enum(ASK_EFFORTS).optional(),
      },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ question, effort }, extra) => {
      try {
        const answer = await askWeb(question, { ...(effort ? { effort } : {}), signal: extra.signal });
        return { content: [{ type: "text", text: answer }] };
      } catch (error) {
        return { isError: true, content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }] };
      }
    },
  );
  await server.connect(new StdioServerTransport());
}
