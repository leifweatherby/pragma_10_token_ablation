/**
 * Tokenizer utilities for decoding n-gram token IDs to human-readable text.
 */

interface VocabData {
  model: string;
  vocab_size: number;
  special_tokens: {
    bos_token: string | null;
    eos_token: string | null;
    unk_token: string | null;
    pad_token: string | null;
    bos_token_id: number | null;
    eos_token_id: number | null;
    unk_token_id: number | null;
    pad_token_id: number | null;
  };
  vocab: Record<string, string>; // token_id -> token_string
}

// Multi-vocab cache structure
const vocabCache = new Map<string, VocabData>();  // vocab filename -> VocabData
const vocabPromises = new Map<string, Promise<VocabData>>();  // filename -> Promise
let vocabIndexCache: { [modelName: string]: string } | null = null;
let vocabIndexPromise: Promise<{ [modelName: string]: string }> | null = null;

/**
 * Load the vocab index mapping models to vocab files
 */
async function loadVocabIndex(): Promise<{ [modelName: string]: string }> {
  if (vocabIndexCache) {
    return vocabIndexCache;
  }

  if (vocabIndexPromise) {
    return vocabIndexPromise;
  }

  vocabIndexPromise = fetch('/vocab-index.json')
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load vocab-index.json: ${response.statusText}`);
      }
      return response.json();
    })
    .then((data) => {
      vocabIndexCache = data;
      return data;
    });

  return vocabIndexPromise;
}

/**
 * Get vocab filename for a given model name
 */
async function getVocabFilename(modelName: string): Promise<string> {
  const index = await loadVocabIndex();
  const filename = index[modelName];

  if (!filename) {
    throw new Error(
      `No vocabulary file found for model: ${modelName}. ` +
      `Please run 'pixi r app-setup' to regenerate vocabulary files.`
    );
  }

  return filename;
}

/**
 * Load vocabulary from vocab.json
 */
export async function loadVocab(): Promise<VocabData> {
  // For backwards compatibility, load vocab.json
  const filename = 'vocab.json';

  if (vocabCache.has(filename)) {
    return vocabCache.get(filename)!;
  }

  if (vocabPromises.has(filename)) {
    return vocabPromises.get(filename)!;
  }

  const promise = fetch('/vocab.json')
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load vocab.json: ${response.statusText}`);
      }
      return response.json();
    })
    .then((data: VocabData) => {
      vocabCache.set(filename, data);
      vocabPromises.delete(filename);
      return data;
    })
    .catch((error) => {
      vocabPromises.delete(filename);
      throw error;
    });

  vocabPromises.set(filename, promise);
  return promise;
}

/**
 * Load vocabulary for a specific model
 * @param modelName - HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
 */
export async function loadVocabForModel(modelName: string): Promise<VocabData> {
  const filename = await getVocabFilename(modelName);

  // Check cache
  if (vocabCache.has(filename)) {
    return vocabCache.get(filename)!;
  }

  // Check if already loading
  if (vocabPromises.has(filename)) {
    return vocabPromises.get(filename)!;
  }

  // Load vocab file
  const promise = fetch(`/${filename}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(
          `Failed to load ${filename}: ${response.statusText}\n` +
          `Model: ${modelName}\n` +
          `Please run 'pixi r app-setup' to regenerate vocabulary files.`
        );
      }
      return response.json();
    })
    .then((data: VocabData) => {
      vocabCache.set(filename, data);
      vocabPromises.delete(filename);
      return data;
    })
    .catch((error) => {
      vocabPromises.delete(filename);
      throw error;
    });

  vocabPromises.set(filename, promise);
  return promise;
}

/**
 * Decode byte-level BPE tokens to proper Unicode characters.
 * Qwen tokenizers use byte-level encoding where bytes are represented as Unicode chars.
 */
function bytesToUnicode(): Map<number, string> {
  // Standard byte-to-unicode mapping used by GPT-2/GPT-3/Qwen tokenizers
  const bs: number[] = [
    ...Array.from({ length: 33 }, (_, i) => i + 33), // '!' to '~'
    ...Array.from({ length: 94 }, (_, i) => i + 33),
    ...Array.from({ length: 172 }, (_, i) => i + 161),
  ].slice(0, 256);

  const cs: number[] = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const byteToChar = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) {
    byteToChar.set(bs[i], String.fromCharCode(cs[i]));
  }

  return byteToChar;
}

/**
 * Reverse the byte-to-unicode mapping for decoding
 */
function getCharToByte(): Map<string, number> {
  const byteToChar = bytesToUnicode();
  const charToByte = new Map<string, number>();
  byteToChar.forEach((char, byte) => {
    charToByte.set(char, byte);
  });
  return charToByte;
}

const charToByte = getCharToByte();

/**
 * Decode a single token string to bytes and then to UTF-8 text
 */
function decodeToken(token: string): string {
  // Convert token characters back to bytes
  const bytes: number[] = [];
  for (const char of token) {
    const byte = charToByte.get(char);
    if (byte !== undefined) {
      bytes.push(byte);
    } else {
      // If character is not in mapping, keep it as-is
      return token;
    }
  }

  // Convert bytes to UTF-8 string
  try {
    const uint8Array = new Uint8Array(bytes);
    return new TextDecoder('utf-8').decode(uint8Array);
  } catch (e) {
    // If decoding fails, return original token
    return token;
  }
}

/**
 * Decode token IDs using the vocabulary for a specific model
 * @param tokenIds - Array of token IDs
 * @param modelName - Model name to use for vocabulary lookup
 * @returns Decoded text string
 */
export async function decodeTokenIdsForModel(
  tokenIds: number[],
  modelName: string
): Promise<string> {
  const vocabData = await loadVocabForModel(modelName);

  const tokens: string[] = [];
  for (const tokenId of tokenIds) {
    const token = vocabData.vocab[tokenId.toString()];
    if (token !== undefined) {
      const decoded = decodeToken(token);
      tokens.push(decoded);
    } else {
      tokens.push(`<unk:${tokenId}>`);
    }
  }

  return tokens.join('');
}

/**
 * Decode multiple n-grams for a specific model
 */
export async function decodeNgramsForModel(
  ngrams: number[][],
  modelName: string
): Promise<string[]> {
  const vocab = await loadVocabForModel(modelName);

  // Decode all n-grams with the loaded vocab
  return Promise.all(
    ngrams.map(async (ngram) => {
      const tokens: string[] = [];
      for (const tokenId of ngram) {
        const token = vocab.vocab[tokenId.toString()];
        if (token !== undefined) {
          const decoded = decodeToken(token);
          tokens.push(decoded);
        } else {
          tokens.push(`<unk:${tokenId}>`);
        }
      }
      return tokens.join('');
    })
  );
}

/**
 * Decode an array of token IDs to human-readable text
 * @param tokenIds - Array of token IDs
 * @param vocab - Optional vocabulary data (will load if not provided)
 * @returns Decoded text string
 */
export async function decodeTokenIds(
  tokenIds: number[],
  vocab?: VocabData
): Promise<string> {
  const vocabData = vocab || (await loadVocab());

  const tokens: string[] = [];
  for (const tokenId of tokenIds) {
    const token = vocabData.vocab[tokenId.toString()];
    if (token !== undefined) {
      // Decode byte-level BPE token to proper Unicode
      const decoded = decodeToken(token);
      tokens.push(decoded);
    } else {
      // Unknown token
      tokens.push(`<unk:${tokenId}>`);
    }
  }

  return tokens.join('');
}

/**
 * Decode multiple n-grams at once (for better performance)
 */
export async function decodeNgrams(ngrams: number[][]): Promise<string[]> {
  const vocab = await loadVocab();
  return Promise.all(ngrams.map((ngram) => decodeTokenIds(ngram, vocab)));
}

/**
 * Get token ID to string mapping (for debugging)
 */
export async function getTokenString(tokenId: number): Promise<string> {
  const vocab = await loadVocab();
  return vocab.vocab[tokenId.toString()] || `<unk:${tokenId}>`;
}
