# EmojiBot

EmojiBot suggests the emoji that best fits a message. Type a sentence and a
bidirectional LSTM scores 43 emoji classes, all locally in the browser.

- **Live demo:** https://emoji-bot-iota.vercel.app
- **Data dashboards:** [Tableau Public](https://public.tableau.com/views/Book2_16990836192980/Dashboard1)

The deployed site is completely static. There is no prediction API and no
request containing your message ever leaves the page. The original PyTorch
checkpoint is exported to a compact int8 binary that the browser downloads
once and runs with plain JavaScript.

## How it works

1. **Tokenize.** The message is lowercased and split into words, URLs,
   mentions, hashtags, emoticons, emoji sequences, and punctuation using a
   JavaScript port of the relevant parts of NLTK's `TweetTokenizer`.
2. **Embed.** Each token is mapped to a 400-dimensional embedding. The
   embedding table is stored as per-row symmetric int8 values with float32
   scales, reducing the model from 122 MB to about 31 MB.
3. **Encode.** A bidirectional LSTM (hidden size 128) reads the sequence in
   both directions. The final forward and backward hidden states are
   concatenated.
4. **Classify.** A linear layer produces logits for 43 emoji labels, and a
   softmax turns them into probabilities. The UI shows the top pick plus two
   alternatives.

Model architecture:

| Component | Shape |
| --- | --- |
| Embedding | 74,976 x 400 |
| LSTM | 400 -> 128, bidirectional |
| Linear | 256 -> 43 |

## Project structure

```text
.
├── public/
│   ├── index.html          # Landing page and chat UI
│   ├── styles.css
│   ├── app.js              # UI, model loading, progress
│   ├── inference.js        # Tokenizer, binary parser, LSTM forward pass
│   └── model/v1/           # Exported browser model and metadata
├── scripts/
│   ├── export_web_model.py # Checkpoint -> quantized browser model
│   └── test_web_inference.mjs
├── app.py                  # Optional Flask API
├── server.js               # Dependency-free local static server
└── vercel.json
```

## Run locally

Node 18 or newer is required for the local server and tests.

```sh
npm start
```

Then open http://localhost:3000. The first load downloads the 31 MB model;
later loads use the browser cache. You can also serve the folder directly:

```sh
python3 -m http.server 3000 --directory public
```

## Test

```sh
npm test
```

This loads the exported binary in Node and checks tokenization plus sample
predictions against values produced by the original PyTorch model.

## Re-export the model

The checkpoint is stored with Git LFS.

```sh
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_web_model.py
```

The script writes `public/model/v1/{model.bin,model.json,vocab.json,emojis.json}`,
prints the SHA-256 digest, and compares the quantized model with the original
on random inputs.

## Optional Flask API

The static site does not use Flask. To experiment with the original server:

```sh
pip install -r requirements.txt
python app.py
```

`POST /predict` accepts `{"message": "This is hilarious"}` and returns the
predicted emoji plus the top three probabilities. `GET /health` reports the
device in use.

## Deploy

`vercel.json` sets `public` as the output directory and gives the versioned
model files an immutable cache policy. Import the repository in Vercel, or run:

```sh
npx vercel --prod
```

No environment variables or build step are required.

## Data dashboards

The page embeds two Tableau Public dashboards:

- **Emoji language patterns:** top words and distinctive terms per emoji.
- **Emoji similarity:** how emojis group together by context.

## Limitations

- The model was trained on tweets, so it reflects the emoji conventions and
  biases of that dataset.
- Only 43 emoji classes are supported.
- Sarcasm and long, nuanced messages are hard; treat the output as a suggestion.
- The JavaScript tokenizer is a close approximation of NLTK's tokenizer, not a
  byte-for-byte port of every edge case.

## Team

YuChen Liao, ChingYuan Kung, HuiChun Mo, Weilong Shen, YuehChang Yang,
Lingfeng Zhang.

Georgia Tech CSE 6242, Data and Visual Analytics.

## License

MIT
