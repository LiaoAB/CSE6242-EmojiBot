"""Optional Flask API for EmojiBot.

The deployed site does not need this server: inference runs in the browser.
This is kept for local API experiments and is not required by the static site.

    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import __main__
import json
import os
from pathlib import Path

import nltk
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MAX_TOKENS = 50

app = Flask(__name__)
CORS(app)


class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, drop_out, bidirectional=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop_out,
        )
        self.linear = torch.nn.Linear(2 * hidden_dim * num_layers, 43)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.linear(hidden)


def load_json(name: str):
    return json.loads((BASE_DIR / name).read_text())


def load_checkpoint(path: Path, target_device: torch.device):
    """The checkpoint was pickled as ``__main__.LSTMModel``.

    Expose this module's class under that name while unpickling so the file
    also loads when app.py is imported instead of run directly.
    """
    had_class = hasattr(__main__, "LSTMModel")
    original_class = getattr(__main__, "LSTMModel", None)
    __main__.LSTMModel = LSTMModel
    try:
        return torch.load(path, map_location=target_device, weights_only=False)
    finally:
        if had_class:
            __main__.LSTMModel = original_class
        else:
            del __main__.LSTMModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_checkpoint(BASE_DIR / "emoji_model.pth", device)
model.eval()
vocab = load_json("train_vocab.json")
label_to_emoji = load_json("label_to_emoji.json")
tokenizer = nltk.tokenize.TweetTokenizer()


def get_prediction(text: str) -> dict:
    ids = []
    for token in tokenizer.tokenize(text.lower()):
        if token in vocab:
            ids.append(vocab[token])
        elif token.startswith("#"):
            ids.append(vocab["UNK_HASHTAG"])
        elif token.startswith("@"):
            ids.append(vocab["UNK_MENTION"])
        else:
            ids.append(vocab["UNK_WORD"])
        if len(ids) >= MAX_TOKENS:
            break

    if not ids:
        ids.append(vocab["UNK_WORD"])

    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        probabilities = torch.softmax(logits, dim=1)[0]

    top = torch.topk(probabilities, k=min(3, len(probabilities)))
    predictions = [
        {"emoji": label_to_emoji[str(index.item())], "probability": round(score.item(), 6)}
        for score, index in zip(top.values, top.indices)
    ]
    return {"reply": predictions[0]["emoji"], "predictions": predictions}


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    return jsonify(get_prediction(message))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
