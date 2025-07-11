/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { createContentGenerator, AuthType } from './contentGenerator.js';
import { OpenAICompatibleContentGenerator } from './openAICompatibleContentGenerator.js';

vi.mock('./openAICompatibleContentGenerator.js');

describe('contentGenerator', () => {
  it('should create an OpenAICompatibleContentGenerator', async () => {
    const mockGenerator = {} as unknown as OpenAICompatibleContentGenerator;
    vi.mocked(OpenAICompatibleContentGenerator).mockImplementation(
      () => mockGenerator,
    );
    const generator = await createContentGenerator({
      model: 'test-model',
      authType: AuthType.USE_LOCAL_LLM,
    });
    expect(OpenAICompatibleContentGenerator).toHaveBeenCalled();
    expect(generator).toBe(mockGenerator);
  });
});
