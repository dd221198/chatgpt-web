import { afterEach, expect, test } from "bun:test";
import type { ProviderAdapter } from "../src/adapters/base";
import { chatGptTurnExecutionKey, chatGptTurnSessions } from "../src/adapters/chatgpt-web/turn-execution";
import { askRoute, askWeb } from "../src/ask";
import { defaultConfig } from "../src/config";
import { responseRequest } from "../src/server";

afterEach(() => {
  chatGptTurnSessions.clear();
});

const solConfig = {
  ...defaultConfig("browser-only"),
  solAvailable: true,
  extraHighAvailable: false,
  proAvailable: false,
};

test("ask sends one tool-free Codex-shaped turn and returns only the final answer", async () => {
  let seen: Record<string, unknown> | undefined;
  const factory = (): ProviderAdapter => ({
    name: "ask-test",
    async runTurn(parsed, _incoming, emit) {
      // Throws when the synthetic native turn envelope is incomplete.
      chatGptTurnExecutionKey(parsed);
      seen = {
        model: parsed.modelId,
        reasoning: parsed.options.reasoning,
        tools: parsed.context.tools ?? [],
        question: parsed.context.messages.at(-1)?.content,
      };
      emit({ type: "text_delta", text: "Searching the web.", phase: "commentary" });
      emit({ type: "text_delta", text: "Yes. Source: https://example.com/odoo", phase: "final_answer" });
      emit({ type: "done", stopReason: "stop", endTurn: true });
    },
  });
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: req => responseRequest(req, solConfig, factory, { rememberState: false }),
  });
  try {
    const answer = await askWeb("Does Odoo 19 support X?", {
      effort: "medium",
      config: { ...solConfig, port: server.port! },
    });
    expect(answer).toBe("Yes. Source: https://example.com/odoo");
    expect(seen).toEqual({
      model: "gpt-5.6-sol",
      reasoning: "medium",
      tools: [],
      question: "Does Odoo 19 support X?",
    });
  } finally {
    server.stop(true);
  }
});

test("ask surfaces a failed ChatGPT turn instead of returning partial text", async () => {
  const factory = (): ProviderAdapter => ({
    name: "ask-failure-test",
    async runTurn(_parsed, _incoming, emit) {
      emit({ type: "text_delta", text: "partial", phase: "final_answer" });
      emit({ type: "error", message: "ChatGPT rate limit reached", status: 429, retryable: false });
    },
  });
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: req => responseRequest(req, solConfig, factory, { rememberState: false }),
  });
  try {
    await expect(askWeb("question", { config: { ...solConfig, port: server.port! } }))
      .rejects.toThrow("ChatGPT rate limit reached");
  } finally {
    server.stop(true);
  }
});

test("ask maps effort to explicit account routes and refuses unsupported setups", () => {
  expect(askRoute(solConfig)).toEqual({ model: "chatgpt-web/gpt-5.6-sol", reasoning: "high" });
  expect(askRoute(solConfig, "instant")).toEqual({ model: "chatgpt-web/gpt-5.6-sol-instant", reasoning: "low" });
  expect(() => askRoute(solConfig, "xhigh")).toThrow("does not support effort");
  expect(() => askRoute(solConfig, "pro")).toThrow("not available for this account");
  expect(askRoute({ ...solConfig, proAvailable: true }, "pro"))
    .toEqual({ model: "chatgpt-web/gpt-6-pro", reasoning: "max" });

  const lunaConfig = { ...solConfig, solAvailable: false };
  expect(askRoute(lunaConfig)).toEqual({ model: "chatgpt-web/gpt-5.6-luna", reasoning: "medium" });
  expect(askRoute(lunaConfig, "instant")).toEqual({ model: "chatgpt-web/gpt-5.6-luna", reasoning: "low" });
  expect(() => askRoute(lunaConfig, "high")).toThrow("does not support effort");

  expect(() => askRoute({ ...solConfig, mode: "full" })).toThrow("Browser-only");
  expect(() => askRoute({ ...solConfig, browserInteractionMode: "manual" })).toThrow("Zero Risk");
});
