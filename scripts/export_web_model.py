#!/usr/bin/env python3
"""Export the EmojiBot model for fully client-side inference.

The trained checkpoint stores a full PyTorch module. This script:

1. Loads the checkpoint and extracts the weights.
2. Quantizes the embedding table to int8 with per-row scales
   (the embedding is ~98% of the model size).
3. Writes a compact binary blob + JSON metadata into public/model/v1/.
4. Checks the quantized model against the original PyTorch model.

Usage:
    python scripts/export_web_model.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "emoji_model.pth"
VOCAB_PATH = ROOT / "train_vocab.json"
LABELS_PATH = ROOT / "label_to_emoji.json"
OUT_DIR = ROOT / "public" / "model" / "v1"

MAGIC = b"EMOJI01\n"
MAX_TOKENS = 50


class LSTMModel(torch.nn.Module):
    """Must match the class used when the checkpoint was saved."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        drop_out: float,
        bidirectional: bool = True,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.linear(hidden)


def align4(value: int) -> int:
    return (value + 3) & ~3


def write_binary(sections: list[tuple[str, np.ndarray, str]]) -> tuple[bytes, dict]:
    """Serialize sections into one aligned binary blob and return offsets."""
    offsets: dict[str, int] = {}
    lengths: dict[str, int] = {}
    parts: list[bytes] = [MAGIC]
    cursor = len(MAGIC)

    for name, array, dtype in sections:
        if array.dtype != np.dtype(dtype):
            array = array.astype(dtype)
        data = array.tobytes(order="C")
        cursor = align4(cursor)
        padding = cursor - sum(len(part) for part in parts)
        if padding:
            parts.append(b"\x00" * padding)
        offsets[name] = cursor
        lengths[name] = array.size
        parts.append(data)
        cursor += len(data)

    return b"".join(parts), {"offsets": offsets, "lengths": lengths}


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"Missing checkpoint: {CHECKPOINT}")

    vocab = json.loads(VOCAB_PATH.read_text())
    labels = json.loads(LABELS_PATH.read_text())

    # The checkpoint contains a pickled module, so weights_only must be False.
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.eval()

    state = model.state_dict()
    embedding = state["embedding.weight"].detach().cpu().numpy().astype(np.float32)
    vocab_size, embedding_dim = embedding.shape
    hidden_dim = state["lstm.weight_hh_l0"].shape[1]
    num_labels = state["linear.weight"].shape[0]
    num_layers = 1

    if vocab_size != len(vocab):
        raise SystemExit(f"Vocab mismatch: {vocab_size} rows vs {len(vocab)} tokens")
    if num_labels != len(labels):
        raise SystemExit(f"Label mismatch: {num_labels} rows vs {len(labels)} emojis")

    # Per-row symmetric int8 quantization. Each row has its own scale, which
    # keeps the quantization error far below a single global scale.
    row_max = np.max(np.abs(embedding), axis=1)
    scales = row_max / 127.0
    scales[scales == 0] = 1.0
    quantized = np.clip(np.round(embedding / scales[:, None]), -127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scales[:, None]

    def weights(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w_ih = state[f"lstm.weight_ih_{prefix}"].detach().cpu().numpy().astype(np.float32)
        w_hh = state[f"lstm.weight_hh_{prefix}"].detach().cpu().numpy().astype(np.float32)
        bias = (
            state[f"lstm.bias_ih_{prefix}"].detach().cpu().numpy()
            + state[f"lstm.bias_hh_{prefix}"].detach().cpu().numpy()
        ).astype(np.float32)
        return w_ih, w_hh, bias

    fwd_w_ih, fwd_w_hh, fwd_bias = weights("l0")
    bwd_w_ih, bwd_w_hh, bwd_bias = weights("l0_reverse")
    linear_w = state["linear.weight"].detach().cpu().numpy().astype(np.float32)
    linear_b = state["linear.bias"].detach().cpu().numpy().astype(np.float32)

    sections = [
        ("embedding", quantized, "int8"),
        ("embeddingScales", scales.astype(np.float32), "<f4"),
        ("lstmForwardWih", fwd_w_ih, "<f4"),
        ("lstmForwardWhh", fwd_w_hh, "<f4"),
        ("lstmForwardBias", fwd_bias, "<f4"),
        ("lstmBackwardWih", bwd_w_ih, "<f4"),
        ("lstmBackwardWhh", bwd_w_hh, "<f4"),
        ("lstmBackwardBias", bwd_bias, "<f4"),
        ("linearWeight", linear_w, "<f4"),
        ("linearBias", linear_b, "<f4"),
    ]

    blob, layout = write_binary(sections)
    sha256 = hashlib.sha256(blob).hexdigest()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model.bin").write_bytes(blob)

    meta = {
        "version": 1,
        "magic": MAGIC.decode().strip(),
        "vocabSize": int(vocab_size),
        "embeddingDim": int(embedding_dim),
        "hiddenDim": int(hidden_dim),
        "numLayers": int(num_layers),
        "numLabels": int(num_labels),
        "maxTokens": MAX_TOKENS,
        "embeddingQuantization": "per-row symmetric int8",
        "sizeBytes": len(blob),
        "sha256": sha256,
        **layout,
    }
    (OUT_DIR / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT_DIR / "vocab.json").write_text(json.dumps(vocab, separators=(",", ":")))
    (OUT_DIR / "emojis.json").write_text(json.dumps(labels, separators=(",", ":")))

    # ---------------- Validation ----------------
    # Swap in the dequantized embedding and compare against the original model.
    quantized_model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    quantized_model.eval()
    with torch.no_grad():
        quantized_model.embedding.weight.copy_(torch.from_numpy(dequantized))

    generator = torch.Generator().manual_seed(7)
    total = 200
    agree = 0
    max_logit_diff = 0.0
    with torch.no_grad():
        for _ in range(total):
            length = int(torch.randint(1, MAX_TOKENS + 1, (1,), generator=generator).item())
            ids = torch.randint(0, vocab_size, (1, length), generator=generator)
            original = model(ids)
            quantized_out = quantized_model(ids)
            if original.argmax(dim=1).item() == quantized_out.argmax(dim=1).item():
                agree += 1
            max_logit_diff = max(
                max_logit_diff,
                float((original - quantized_out).abs().max().item()),
            )

    size_mb = len(blob) / (1024 * 1024)
    print(f"Wrote {OUT_DIR / 'model.bin'} ({size_mb:.1f} MiB)")
    print(f"sha256: {sha256}")
    print(f"Quantization agreement: {agree}/{total} ({agree / total:.1%})")
    print(f"Max logit difference: {max_logit_diff:.6f}")

    if agree < total:
        print("WARNING: quantization changed some predictions")


if __name__ == "__main__":
    main()
