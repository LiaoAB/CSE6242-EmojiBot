import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createPredictor, tokenizeTweet } from "../public/inference.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const modelDir = join(root, "public", "model", "v1");

const readJson = (name) => JSON.parse(readFileSync(join(modelDir, name), "utf8"));
const meta = readJson("model.json");
const vocab = readJson("vocab.json");
const labels = readJson("emojis.json");
const buffer = readFileSync(join(modelDir, "model.bin"));
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

const predictor = createPredictor({ buffer: arrayBuffer, meta, vocab, labels });

const tokenCases = [
  ["This is hilarious lol", ["this", "is", "hilarious", "lol"]],
  ["Thank you so much!", ["thank", "you", "so", "much", "!"]],
  ["Let's party tonight", ["let's", "party", "tonight"]],
  ["@user this is #awesome", ["@user", "this", "is", "#awesome"]],
  ["I can't believe this...", ["i", "can't", "believe", "this", "..."]],
  ["hello :-) <3", ["hello", ":-)", "<3"]],
  ["Check https://example.com/a?b=1 now", ["check", "https://example.com/a?b=1", "now"]],
  ["Family 👨‍👩‍👧‍👦 time", ["family", "👨‍👩‍👧‍👦", "time"]]
];

let failures = 0;
for (const [input, expected] of tokenCases) {
  const actual = tokenizeTweet(input);
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  console.log(`${ok ? "PASS" : "FAIL"} tokenize ${JSON.stringify(input)}`);
  if (!ok) {
    failures += 1;
    console.log(`  expected: ${JSON.stringify(expected)}`);
    console.log(`  actual:   ${JSON.stringify(actual)}`);
  }
}

const predictionCases = [
  ["This is hilarious lol", "🤣"],
  ["That is absolutely amazing", "😍"],
  ["Thank you so much!", "🥰"],
  ["Let's party tonight", "🎉"],
  ["I am so angry right now", "😡"],
  ["Good morning everyone", "☀️"],
  ["This food is delicious", "😋"],
  ["I miss you", "😢"],
  ["You are the best", "❤️"],
  ["I just got the job!", "🥳"]
];

for (const [input, expected] of predictionCases) {
  const { predictions, tokens } = predictor.predict(input);
  const actual = predictions[0]?.emoji;
  const ok = actual === expected;
  console.log(
    `${ok ? "PASS" : "FAIL"} predict ${JSON.stringify(input)} -> ${actual} ` +
      `(${(predictions[0]?.probability * 100).toFixed(1)}%) tokens=${JSON.stringify(tokens)}`
  );
  if (!ok) {
    failures += 1;
    console.log(`  expected: ${expected}`);
    console.log(`  top3: ${predictions.map((p) => `${p.emoji} ${(p.probability * 100).toFixed(1)}%`).join(", ")}`);
  }
}

if (failures) {
  console.error(`\n${failures} failure(s)`);
  process.exitCode = 1;
} else {
  console.log("\nAll checks passed");
}
