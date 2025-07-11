/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
  FinishReason,
  ContentListUnion,
  FunctionCall,
  Candidate,
  GenerateContentResponseUsageMetadata,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

export interface OpenAIConfig {
  endpoint: string;
  model: string;
  apiKey?: string;
}

// Create a custom response class that matches the expected interface
class OpenAIGenerateContentResponse {
  candidates: Candidate[];
  private _functionCalls: FunctionCall[] | undefined;
  usageMetadata?: GenerateContentResponseUsageMetadata;

  constructor(
    candidates: Candidate[],
    functionCalls?: FunctionCall[],
    usageMetadata?: GenerateContentResponseUsageMetadata,
  ) {
    this.candidates = candidates;
    this._functionCalls = functionCalls;
    this.usageMetadata = usageMetadata;
  }

  get text(): string {
    const parts = this.candidates?.[0]?.content?.parts;
    if (!parts) {
      return '';
    }
    return parts
      .map((part: Part & { text?: string }) => part.text ?? '')
      .filter((text: string) => text.length > 0)
      .join('');
  }

  get data(): string {
    return '';
  }

  get functionCalls(): FunctionCall[] {
    return this._functionCalls ?? [];
  }

  get executableCode(): string {
    return '';
  }

  get codeExecutionResult(): string {
    return '';
  }
}

export class OpenAICompatibleContentGenerator implements ContentGenerator {
  constructor(private config: OpenAIConfig) {}

  private convertToOpenAIMessages(
    contents: ContentListUnion,
  ): Array<Record<string, unknown>> {
    // Handle string input
    if (typeof contents === 'string') {
      return [{ role: 'user', content: contents }];
    }

    // Handle single Content object
    if (!Array.isArray(contents)) {
      const content = contents as Content;
      const role = content.role === 'model' ? 'assistant' : content.role;
      const parts = content.parts || [];

      const textParts = parts
        .filter((part: Part) => 'text' in part)
        .map((part: Part & { text?: string }) => part.text ?? '')
        .join('\n');

      return [
        {
          role,
          content: textParts,
        },
      ];
    }

    // Handle array of Content objects
    return contents.map((content: Content) => {
      const role = content.role === 'model' ? 'assistant' : content.role;
      const parts = content.parts || [];

      // Combine all text parts into a single message
      const textParts = parts
        .filter((part: Part) => 'text' in part)
        .map((part: Part & { text?: string }) => part.text ?? '')
        .join('\n');

      return {
        role,
        content: textParts,
      };
    });
  }

  private convertFromOpenAIResponse(
    openAIResponse: unknown,
  ): GenerateContentResponse {
    const choice = (
      openAIResponse as {
        choices?: Array<{
          message?: {
            content?: string;
            tool_calls?: Array<{
              id: string;
              function: { name: string; arguments?: string };
            }>;
          };
          finish_reason?: string;
        }>;
      }
    ).choices?.[0];
    const messageContent = choice?.message?.content || '';
    const toolCalls = choice?.message?.tool_calls as
      | Array<{ id: string; function: { name: string; arguments?: string } }>
      | undefined;

    if (!choice) {
      return new OpenAIGenerateContentResponse([]);
    }

    const parts: Part[] = [];
    if (messageContent) {
      parts.push({ text: messageContent });
    }
    let functionCalls: FunctionCall[] | undefined;
    if (toolCalls && toolCalls.length > 0) {
      functionCalls = toolCalls.map((tc) => {
        let argsObj: Record<string, unknown> = {};
        if (tc.function.arguments) {
          try {
            argsObj = JSON.parse(tc.function.arguments);
          } catch {
            console.warn(
              `Failed to parse function arguments for ${tc.function.name}`,
            );
          }
        }
        return {
          id: tc.id,
          name: tc.function.name,
          args: argsObj,
          isClientInitiated: false,
        };
      });
      for (const fc of functionCalls) {
        parts.push({ functionCall: fc });
      }
    }
    const content: Content = {
      role: 'model',
      parts,
    };

    const candidates = [
      {
        content,
        index: 0,
        finishReason:
          choice.finish_reason === 'stop'
            ? FinishReason.STOP
            : FinishReason.OTHER,
      },
    ];

    const usage = (
      openAIResponse as {
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
          total_tokens?: number;
        };
      }
    ).usage;
    const usageMetadata = usage
      ? {
          promptTokenCount: usage.prompt_tokens,
          candidatesTokenCount: usage.completion_tokens,
          totalTokenCount: usage.total_tokens,
        }
      : undefined;

    return new OpenAIGenerateContentResponse(
      candidates,
      functionCalls,
      usageMetadata,
    );
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.convertToOpenAIMessages(request.contents);

    // Add system instruction if provided
    if (request.config?.systemInstruction) {
      messages.unshift({
        role: 'system',
        content:
          typeof request.config.systemInstruction === 'string'
            ? request.config.systemInstruction
            : (request.config.systemInstruction as { text?: string }).text ||
              '',
      });
    }

    const openAIRequest: Record<string, unknown> = {
      model: this.config.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      top_p: request.config?.topP,
      stream: false,
    };

    if (request.config?.tools && request.config.tools.length > 0) {
      const tools = request.config.tools.flatMap((t) =>
        'functionDeclarations' in t && Array.isArray(t.functionDeclarations)
          ? t.functionDeclarations
          : [],
      );
      if (tools.length > 0) {
        openAIRequest['tools'] = tools.map((fd) => ({
          type: 'function',
          function: {
            name: fd.name,
            description: fd.description,
            parameters: fd.parameters,
          },
        }));
      }
    }

    try {
      const response = await fetch(`${this.config.endpoint}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && {
            Authorization: `Bearer ${this.config.apiKey}`,
          }),
        },
        body: JSON.stringify(openAIRequest),
      });

      if (!response.ok) {
        throw new Error(
          `OpenAI API error: ${response.status} ${response.statusText}`,
        );
      }

      const openAIResponse = await response.json();
      return this.convertFromOpenAIResponse(openAIResponse);
    } catch (error) {
      console.error('Error calling OpenAI compatible API:', error);
      throw error;
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const generator = this._generateContentStream(request);
    return generator;
  }

  private async *_generateContentStream(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    if (request.config?.tools && request.config.tools.length > 0) {
      yield await this.generateContent(request);
      return;
    }
    const messages = this.convertToOpenAIMessages(request.contents);

    // Add system instruction if provided
    if (request.config?.systemInstruction) {
      messages.unshift({
        role: 'system',
        content:
          typeof request.config.systemInstruction === 'string'
            ? request.config.systemInstruction
            : (request.config.systemInstruction as { text?: string }).text ||
              '',
      });
    }

    const openAIRequest: Record<string, unknown> = {
      model: this.config.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      top_p: request.config?.topP,
      stream: true,
    };

    try {
      const response = await fetch(`${this.config.endpoint}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && {
            Authorization: `Bearer ${this.config.apiKey}`,
          }),
        },
        body: JSON.stringify(openAIRequest),
      });

      if (!response.ok) {
        throw new Error(
          `OpenAI API error: ${response.status} ${response.statusText}`,
        );
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last line if it's incomplete
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const chunk = JSON.parse(data);
              const delta = chunk.choices?.[0]?.delta;

              if (delta?.content) {
                const finishReason = chunk.choices?.[0]?.finish_reason;
                const candidates = [
                  {
                    content: {
                      role: 'model',
                      parts: [{ text: delta.content }],
                    },
                    index: 0,
                    finishReason:
                      finishReason === 'stop' ? FinishReason.STOP : undefined,
                  },
                ];

                yield new OpenAIGenerateContentResponse(candidates);
              }
            } catch {
              // Ignore parse errors for individual chunks
            }
          }
        }
      }
    } catch (error) {
      console.error('Error calling OpenAI compatible streaming API:', error);
      throw error;
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Simple approximation - you might want to use a proper tokenizer
    let text = '';
    if (typeof request.contents === 'string') {
      text = request.contents;
    } else if (Array.isArray(request.contents)) {
      text = request.contents
        .flatMap((content: Content) => content.parts || [])
        .filter((part: Part) => 'text' in part)
        .map((part: Part & { text?: string }) => part.text ?? '')
        .join(' ');
    } else {
      // Single Content object
      const content = request.contents as Content;
      text =
        content.parts
          ?.filter((part: Part) => 'text' in part)
          .map((part: Part & { text?: string }) => part.text ?? '')
          .join(' ') || '';
    }

    // Rough approximation: 1 token ≈ 4 characters
    const tokenCount = Math.ceil(text.length / 4);

    return {
      totalTokens: tokenCount,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const count = Array.isArray(request.contents) ? request.contents.length : 1;
    return {
      embeddings: Array.from({ length: count }, () => ({
        values: new Array(768).fill(0),
      })),
    };
  }
}
