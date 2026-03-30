"""
Quick test: load a randomly-initialized BPEAutoRegressiveUnet and compare the
model's actual down_gate_samples against the BPE-prescribed boundaries for the
first sequence in the train set.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from bpe_tokenizer.bpe_tokenizer import BPEAutoRegressiveUnet
from data_processing import split_fineweb

SEQ_LEN = 4096  # bytes (including BOS + EOS)


def positions_of(gate_row: torch.Tensor) -> list:
    return gate_row.cpu().nonzero(as_tuple=True)[0].tolist()


def main():
    # ------------------------------------------------------------------ #
    # 1. Load train set and grab the first document
    # ------------------------------------------------------------------ #
    print("Loading train split …")
    train_set, val_set, _ = split_fineweb.get_splits()
    text = train_set[0]["text"]
    print(f"\n--- text (first 200 chars) ---\n{text[:200]!r}\n")

    # ------------------------------------------------------------------ #
    # 2. Build a tiny randomly-initialized model wrapped in BPEAutoRegressiveUnet
    # ------------------------------------------------------------------ #
    print("Building randomly-initialized BPEAutoRegressiveUnet (size=20M, arch=random) …")
    model = BPEAutoRegressiveUnet.from_config(size="20M", architecture="random").to("cuda", dtype=torch.bfloat16)
    model.eval()

    device = torch.device("cuda")

    # ------------------------------------------------------------------ #
    # 3. BPE-prescribed gate tensor
    # ------------------------------------------------------------------ #
    prescribed_gates = model.texts_to_gate_tensors([text], seq_len=SEQ_LEN, device=device)
    bpe_positions = positions_of(prescribed_gates[0])
    print(f"BPE boundaries : {len(bpe_positions)} tokens")
    print(f"  first 20 positions: {bpe_positions[:20]}")

    # ------------------------------------------------------------------ #
    # 4. Run forward() and extract down_gate_samples
    # ------------------------------------------------------------------ #
    bos = 1
    text_bytes = text.encode("utf-8")
    content = list(text_bytes[: SEQ_LEN - 2])
    ids = [bos] + content + [2]          # 2 = EOS
    if len(ids) < SEQ_LEN:
        ids += [0] * (SEQ_LEN - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids, texts=[text])

    model_gate_row = out["down_gate_samples"][0]   # (SEQ_LEN,)
    model_positions = positions_of(model_gate_row)
    print(f"\nModel down_gate_samples: {len(model_positions)} boundaries")
    print(f"  first 20 positions: {model_positions[:20]}")

    # ------------------------------------------------------------------ #
    # 5. Compare
    # ------------------------------------------------------------------ #
    bpe_set   = set(bpe_positions)
    model_set = set(model_positions)

    tp = bpe_set & model_set          # model fires where BPE says to
    fp = model_set - bpe_set          # model fires where BPE says not to
    fn = bpe_set - model_set          # model misses a BPE boundary

    precision = len(tp) / len(model_set) if model_set else float("nan")
    recall    = len(tp) / len(bpe_set)  if bpe_set   else float("nan")

    print(f"\n--- Alignment ---")
    print(f"  TP (correct boundaries) : {len(tp)}")
    print(f"  FP (extra boundaries)   : {len(fp)}")
    print(f"  FN (missed boundaries)  : {len(fn)}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")

    # Show a side-by-side of the first 30 positions
    print("\n--- First 30 positions side-by-side ---")
    print(f"  {'pos':>5}  {'BPE':>5}  {'model':>5}  match")
    all_pos = sorted(bpe_set | model_set)[:30]
    for pos in all_pos:
        in_bpe   = "1" if pos in bpe_set   else "0"
        in_model = "1" if pos in model_set else "0"
        match    = "<--" if in_bpe != in_model else ""
        print(f"  {pos:>5}  {in_bpe:>5}  {in_model:>5}  {match}")

    # ------------------------------------------------------------------ #
    # 6. Show the BPE tokens around the first few mismatches
    # ------------------------------------------------------------------ #
    mismatches = sorted(fp | fn)[:10]
    if mismatches:
        print("\n--- Token context around first mismatches ---")
        for pos in mismatches:
            byte_pos = max(pos - 1, 0)
            snippet  = text_bytes[max(byte_pos - 8, 0) : byte_pos + 8]
            kind = "FP (extra)" if pos in fp else "FN (missed)"
            print(f"  pos {pos:4d} [{kind}]  …{snippet!r}…")


if __name__ == "__main__":
    main()
