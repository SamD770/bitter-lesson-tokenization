import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from data_processing import split_fineweb

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "trained_tokenizer")
TARGET_DOWNSAMPLE_RATE = 0.204  # tokens/byte (approximately what we converge on)

VOCAB_LIMIT = 200_000  # give up on current doc count and move to next stage

# Two-stage document counts: sweep vocab at 200k docs first, then 500k when vocab size found
N_DOCS = 500_000


def get_training_iterator(train_set, max_documents=None):
    """Yield text strings from FineWeb train set, optionally capped."""
    for i, example in enumerate(train_set):
        if max_documents is not None and i >= max_documents:
            break
        yield example["text"]


def build_tokenizer():
    """Construct an untrained BPE tokenizer with ByteLevel pre-tokenization."""
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    return tokenizer


def train(tokenizer, train_set, vocab_size, max_documents=None, output_dir=TOKENIZER_DIR):
    """Train the tokenizer on train_set and save to output_dir."""
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        get_training_iterator(train_set, max_documents), trainer=trainer
    )

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    return tokenizer


def measure_compression(tokenizer, val_set):
    """Return mean tokens/byte on val_set (target: ~0.2 tokens/byte = ~5 bytes/token)."""
    rates = []
    for example in val_set:
        text = example["text"]
        n_bytes = len(text.encode("utf-8"))
        # encode() adds BOS/EOS via post_processor; subtract 2 to count only content tokens
        n_tokens = len(tokenizer.encode(text).ids) - 2
        rates.append(n_tokens / n_bytes)
    mean = sum(rates) / len(rates)
    std = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
    return {"mean_tokens_per_byte": mean, "std": std, "n_docs": len(rates)}


def main():
    train_set, val_set, _ = split_fineweb.get_splits()

    print(f"\n  vocab={VOCAB_LIMIT:,}, docs={N_DOCS:,}")
    tokenizer = build_tokenizer()
    tokenizer = train(tokenizer, train_set, vocab_size=VOCAB_LIMIT, max_documents=N_DOCS)

    stats = measure_compression(tokenizer, val_set)
    rate = stats["mean_tokens_per_byte"]
    print(
        f"  → {rate:.4f} tokens/byte (±{stats['std']:.4f})"
        f"  = {1/rate:.2f} bytes/token"
        f"  [target ≤ {TARGET_DOWNSAMPLE_RATE}]"
    )
    print(f"Tokenizer saved to: {TOKENIZER_DIR}")

if __name__ == "__main__":
    main()
