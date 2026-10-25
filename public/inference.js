/**
 * Client-side EmojiBot inference.
 *
 * The PyTorch LSTM is exported to an int8-quantized binary by
 * scripts/export_web_model.py. Everything below runs in the browser:
 * tokenization, embedding lookup, the bidirectional LSTM, and the classifier.
 */

const WORD = String.raw`[\p{L}\p{N}_]`;
const EMOJI = String.raw`\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?(?:\u200D\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?)*`;
const TOKEN_PATTERN = new RegExp(
  [
    String.raw`https?:\/\/\S+`,
    String.raw`\b(?:[a-z0-9-]+\.)+(?:[a-z]{2,13})\/?`,
    String.raw`[\u{1F1E6}-\u{1F1FF}]{2}`,
    EMOJI,
    String.raw`[<>]?[:;=8][\-o*']?[)\]\[(dDpP/:}{@|\\]`,
    String.raw`[)\]\[(dDpP/:}{@|\\][\-o*']?[:;=8][<>]?`,
    String.raw`<\/?3`,
    String.raw`[\-]+>|<[\-]+`,
    String.raw`@${WORD}+`,
    String.raw`#+${WORD}+(?:['’\-_]${WORD}+)*`,
    String.raw`${WORD}+(?:[.+\-]${WORD}+)*@${WORD}+(?:[.\-]${WORD}+)+`,
    String.raw`[+\-]?\d+(?:[,/.:-]\d+)+[+\-]?`,
    String.raw`${WORD}+(?:['’\-_]${WORD}+)*`,
    String.raw`\.{2,}`,
    String.raw`\S`
  ].join("|"),
  "gu"
);

const NAMED_ENTITIES = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " "
};

function decodeEntities(text) {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (match, code) => {
    if (code[0] === "#") {
      const hex = code[1] === "x" || code[1] === "X";
      const value = parseInt(code.slice(hex ? 2 : 1), hex ? 16 : 10);
      return Number.isFinite(value) ? String.fromCodePoint(value) : match;
    }
    return NAMED_ENTITIES[code.toLowerCase()] ?? match;
  });
}

/**
 * A compact port of the parts of NLTK's TweetTokenizer that matter for chat
 * text: URLs, mentions, hashtags, emoticons, emoji sequences, words, and
 * punctuation. The model was trained with lowercased input, so we lowercase
 * before tokenizing.
 */
export function tokenizeTweet(text) {
  const normalized = decodeEntities(String(text)).toLowerCase();
  return [...normalized.matchAll(TOKEN_PATTERN)].map((match) => match[0]);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function readUint32(view, offset) {
  return view.getUint32(offset, true);
}

export function createPredictor({ buffer, meta, vocab, labels }) {
  assert(meta && typeof meta === "object", "Missing model metadata");
  assert(buffer instanceof ArrayBuffer, "Model data must be an ArrayBuffer");

  const bytes = new Uint8Array(buffer);
  const expectedMagic = `${meta.magic}\n`;
  const magic = new TextDecoder().decode(bytes.subarray(0, expectedMagic.length));
  assert(magic === expectedMagic, `Unexpected model format: got ${JSON.stringify(magic)}`);
  assert(bytes.byteLength >= meta.sizeBytes, "Model file is truncated");

  const {
    vocabSize,
    embeddingDim,
    hiddenDim,
    numLabels,
    maxTokens,
    offsets,
    lengths
  } = meta;

  const view = new DataView(buffer);
  const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  assert(littleEndian, "Big-endian devices are not supported");

  const needs = [
    "embedding",
    "embeddingScales",
    "lstmForwardWih",
    "lstmForwardWhh",
    "lstmForwardBias",
    "lstmBackwardWih",
    "lstmBackwardWhh",
    "lstmBackwardBias",
    "linearWeight",
    "linearBias"
  ];
  needs.forEach((name) => {
    assert(Number.isInteger(offsets?.[name]), `Missing offset for ${name}`);
    assert(Number.isInteger(lengths?.[name]), `Missing length for ${name}`);
  });

  const embedding = new Int8Array(buffer, offsets.embedding, lengths.embedding);
  const embeddingScales = new Float32Array(buffer, offsets.embeddingScales, lengths.embeddingScales);
  const lstmForwardWih = new Float32Array(buffer, offsets.lstmForwardWih, lengths.lstmForwardWih);
  const lstmForwardWhh = new Float32Array(buffer, offsets.lstmForwardWhh, lengths.lstmForwardWhh);
  const lstmForwardBias = new Float32Array(buffer, offsets.lstmForwardBias, lengths.lstmForwardBias);
  const lstmBackwardWih = new Float32Array(buffer, offsets.lstmBackwardWih, lengths.lstmBackwardWih);
  const lstmBackwardWhh = new Float32Array(buffer, offsets.lstmBackwardWhh, lengths.lstmBackwardWhh);
  const lstmBackwardBias = new Float32Array(buffer, offsets.lstmBackwardBias, lengths.lstmBackwardBias);
  const linearWeight = new Float32Array(buffer, offsets.linearWeight, lengths.linearWeight);
  const linearBias = new Float32Array(buffer, offsets.linearBias, lengths.linearBias);

  assert(embedding.length === vocabSize * embeddingDim, "Embedding shape mismatch");
  assert(embeddingScales.length === vocabSize, "Embedding scale count mismatch");
  assert(lstmForwardBias.length === 4 * hiddenDim, "LSTM bias shape mismatch");
  assert(linearWeight.length === numLabels * 2 * hiddenDim, "Linear weight shape mismatch");

  const unkWord = vocab.UNK_WORD ?? 0;
  const unkHashtag = vocab.UNK_HASHTAG ?? unkWord;
  const unkMention = vocab.UNK_MENTION ?? unkWord;

  const gateCount = 4 * hiddenDim;
  const x = new Float32Array(embeddingDim);
  const gates = new Float32Array(gateCount);
  const hForward = new Float32Array(hiddenDim);
  const cForward = new Float32Array(hiddenDim);
  const hBackward = new Float32Array(hiddenDim);
  const cBackward = new Float32Array(hiddenDim);
  const hidden = new Float32Array(2 * hiddenDim);
  const logits = new Float32Array(numLabels);

  function textToIds(text) {
    const tokens = tokenizeTweet(text);
    const ids = [];
    for (const token of tokens) {
      if (Object.prototype.hasOwnProperty.call(vocab, token)) {
        ids.push(vocab[token]);
      } else if (token.startsWith("#")) {
        ids.push(unkHashtag);
      } else if (token.startsWith("@")) {
        ids.push(unkMention);
      } else {
        ids.push(unkWord);
      }
      if (ids.length >= maxTokens) break;
    }
    if (ids.length === 0) ids.push(unkWord);
    return { tokens, ids };
  }

  function runLstm(ids, reverse, wih, whh, bias, h, c) {
    for (let step = 0; step < ids.length; step += 1) {
      const id = ids[reverse ? ids.length - 1 - step : step];
      const row = id * embeddingDim;
      const scale = embeddingScales[id];
      for (let i = 0; i < embeddingDim; i += 1) {
        x[i] = embedding[row + i] * scale;
      }

      for (let gate = 0; gate < gateCount; gate += 1) {
        let sum = bias[gate];
        const wihRow = gate * embeddingDim;
        for (let i = 0; i < embeddingDim; i += 1) {
          sum += wih[wihRow + i] * x[i];
        }
        const whhRow = gate * hiddenDim;
        for (let i = 0; i < hiddenDim; i += 1) {
          sum += whh[whhRow + i] * h[i];
        }
        gates[gate] = sum;
      }

      for (let i = 0; i < hiddenDim; i += 1) {
        const inputGate = 1 / (1 + Math.exp(-gates[i]));
        const forgetGate = 1 / (1 + Math.exp(-gates[hiddenDim + i]));
        const cellGate = Math.tanh(gates[2 * hiddenDim + i]);
        const outputGate = 1 / (1 + Math.exp(-gates[3 * hiddenDim + i]));
        c[i] = forgetGate * c[i] + inputGate * cellGate;
        h[i] = outputGate * Math.tanh(c[i]);
      }
    }
  }

  function forward(ids) {
    hForward.fill(0);
    cForward.fill(0);
    hBackward.fill(0);
    cBackward.fill(0);

    runLstm(ids, false, lstmForwardWih, lstmForwardWhh, lstmForwardBias, hForward, cForward);
    runLstm(ids, true, lstmBackwardWih, lstmBackwardWhh, lstmBackwardBias, hBackward, cBackward);

    hidden.set(hForward, 0);
    hidden.set(hBackward, hiddenDim);

    let maxLogit = -Infinity;
    for (let label = 0; label < numLabels; label += 1) {
      let sum = linearBias[label];
      const row = label * 2 * hiddenDim;
      for (let i = 0; i < 2 * hiddenDim; i += 1) {
        sum += linearWeight[row + i] * hidden[i];
      }
      logits[label] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    let total = 0;
    for (let label = 0; label < numLabels; label += 1) {
      logits[label] = Math.exp(logits[label] - maxLogit);
      total += logits[label];
    }
    for (let label = 0; label < numLabels; label += 1) {
      logits[label] /= total;
    }
    return logits;
  }

  function predict(text, topK = 3) {
    const { tokens, ids } = textToIds(text);
    const probabilities = forward(ids);
    const ranked = Array.from(probabilities, (probability, id) => ({
      id,
      emoji: labels[String(id)] ?? "",
      probability
    }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, topK);

    return { tokens, ids, predictions: ranked };
  }

  return {
    predict,
    meta: {
      vocabSize,
      embeddingDim,
      hiddenDim,
      numLabels,
      maxTokens
    }
  };
}
