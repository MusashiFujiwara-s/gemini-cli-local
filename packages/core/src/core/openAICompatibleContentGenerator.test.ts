/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OpenAICompatibleContentGenerator } from './openAICompatibleContentGenerator.js';
import type { Content, Part } from '@google/genai';

const endpoint = 'http://localhost';
const model = 'test-model';

describe('OpenAICompatibleContentGenerator', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('converts functionResponse parts to tool messages', async () => {
    const generator = new OpenAICompatibleContentGenerator({ endpoint, model });

    const content: Content = {
      role: 'tool',
      parts: [
        {
          functionResponse: {
            id: 'call1',
            name: 'toolTest',
            response: { content: [{ text: 'done' }] },
          },
        } as Part,
      ],
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi
        .fn()
        .mockResolvedValue({ choices: [{ message: { content: 'ok' } }] }),
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.fetch = fetchMock as any;

    await generator.generateContent({ model, contents: [content] });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.messages[0]).toEqual({
      role: 'tool',
      content: 'done',
      tool_call_id: 'call1',
    });
  });

  it('converts functionCall parts to assistant tool_call messages', async () => {
    const generator = new OpenAICompatibleContentGenerator({ endpoint, model });

    const content: Content = {
      role: 'model',
      parts: [
        {
          functionCall: {
            id: 'call1',
            name: 'toolTest',
            args: { foo: 'bar' },
            isClientInitiated: false,
          },
        } as Part,
      ],
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi
        .fn()
        .mockResolvedValue({ choices: [{ message: { content: 'ok' } }] }),
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.fetch = fetchMock as any;

    await generator.generateContent({ model, contents: [content] });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.messages[0]).toEqual({
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'call1',
          type: 'function',
          function: { name: 'toolTest', arguments: '{"foo":"bar"}' },
        },
      ],
    });
  });
});
