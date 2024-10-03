export type ChatRole = "assistant" | "user";

export type OpenAIResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | {
      type: "json_schema";
      json_schema: {
        name: string;
        description?: string;
        schema?: any;
        strict?: boolean;
      };
    };

export interface ToolsInterface {
  name: string;
  description?: string;
  schema?: any;
}

export interface ChatGeneratorSettings {
  max_tokens: number;
  messages: { role: ChatRole; content: string }[];
  system?: string;
  temperature?: number;
  stop_token?: string[];
  frequency_penalty?: number;
  presence_penalty?: number;
  top_p?: number;
  n?: number;
  logit_bias?: { [tokenId: string]: number };
  logprobs?: boolean;
  top_logprobs?: number;
  seed?: number;
  service_tier?: "auto" | "default" | null;
  top_k?: number;
  response_format?: OpenAIResponseFormat;
  tools?: ToolsInterface[];
}
