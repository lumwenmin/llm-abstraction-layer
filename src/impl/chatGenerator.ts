import {
  ChatGeneratorSettings,
  OpenAIResponseFormat,
  ToolsInterface,
} from "../models";
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { ChatCompletionTool } from "openai/resources/index.mjs";
import { Tool } from "@anthropic-ai/sdk/resources/messages.mjs";

export class ChatGenerator {
  private max_tokens: number;
  private messages: { role: string; content: string }[];
  private system?: string;
  private temperature?: number;
  private stop_token?: string[];
  private frequency_penalty?: number;
  private presence_penalty?: number;
  private top_p?: number;
  private n?: number;
  private logit_bias?: { [tokenId: string]: number };
  private logprobs?: boolean;
  private top_logprobs?: number;
  private seed?: number;
  private service_tier?: "auto" | "default" | null;
  private top_k?: number;
  private response_format?: OpenAIResponseFormat;
  private tools?: ToolsInterface[];

  constructor(settings: ChatGeneratorSettings) {
    this.max_tokens = settings.max_tokens;
    this.messages = settings.messages;
    this.system = settings.system;
    this.temperature = settings.temperature;
    this.stop_token = settings.stop_token;
    this.frequency_penalty = settings.frequency_penalty;
    this.presence_penalty = settings.presence_penalty;
    this.top_p = settings.top_p;
    this.n = settings.n;
    this.logit_bias = settings.logit_bias;
    this.logprobs = settings.logprobs;
    this.top_logprobs = settings.top_logprobs;
    this.seed = settings.seed;
    this.service_tier = settings.service_tier;
    this.top_k = settings.top_k;
    this.response_format = settings.response_format;
    this.tools = settings.tools;
  }

  async generate(model_name: string): Promise<string> {
    if (model_name.toLowerCase().includes("claude")) {
      return this.generateWithClaude(model_name);
    }

    return this.generateWithOpenAI(model_name);
  }

  private async generateWithClaude(claude_model: string): Promise<any> {
    const llm = new Anthropic();
    const response = await llm.messages.create({
      model: claude_model,
      max_tokens: this.max_tokens,
      messages: this.messages.map((msg) => ({
        role: msg.role as "user" | "assistant",
        content: msg.content,
      })),
      ...(this.system ? { system: this.system } : {}),
      ...(this.temperature ? { temperature: this.temperature / 2 } : {}),
      ...(this.stop_token ? { stop_sequences: this.stop_token } : {}),
      ...(this.top_p ? { top_p: this.top_p } : {}),
      ...(this.top_k ? { top_k: this.top_k } : {}),
      ...(this.tools ? { tools: this.processAnthropicTools() } : {}),
    });

    return response.content[0];
  }

  private async generateWithOpenAI(openai_model: string): Promise<any> {
    const llm = new OpenAI();
    const messages = [
      ...(this.system
        ? [{ role: "system" as const, content: this.system }]
        : []),
      ...this.messages.map((msg) => ({
        role: msg.role as "system" | "user" | "assistant",
        content: msg.content,
      })),
    ];

    const response = await llm.chat.completions.create({
      model: openai_model,
      max_tokens: this.max_tokens,
      messages,
      ...(this.temperature ? { temperature: this.temperature } : {}),
      ...(this.stop_token ? { stop: this.stop_token } : {}),
      ...(this.frequency_penalty
        ? { frequency_penalty: this.frequency_penalty }
        : {}),
      ...(this.presence_penalty
        ? { presence_penalty: this.presence_penalty }
        : {}),
      ...(this.top_p ? { top_p: this.top_p } : {}),
      ...(this.n ? { n: this.n } : {}),
      ...(this.logit_bias ? { logit_bias: this.logit_bias } : {}),
      ...(this.logprobs ? { logprobs: this.logprobs } : {}),
      ...(this.top_logprobs ? { top_logprobs: this.top_logprobs } : {}),
      ...(this.seed ? { seed: this.seed } : {}),
      ...(this.service_tier ? { service_tier: this.service_tier } : {}),
      ...(this.response_format
        ? { response_format: this.response_format }
        : {}),
      ...(this.tools ? { tools: this.processOpenAITools() } : {}),
    });

    return response.choices[0].message.content;
  }

  private processOpenAITools(): ChatCompletionTool[] {
    return this.tools
      ? this.tools.map((tool) => this.createOpenAITool(tool))
      : [];
  }

  private createOpenAITool(tool: ToolsInterface): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.schema,
      },
    };
  }

  private processAnthropicTools(): Tool[] {
    return this.tools
      ? this.tools.map((tool) => this.createAnthropicTool(tool))
      : [];
  }

  private createAnthropicTool(tool: ToolsInterface): Tool {
    return {
      name: tool.name,
      description: tool.description,
      input_schema: tool.schema,
    };
  }
}
