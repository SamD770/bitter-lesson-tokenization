"""
BPEAutoRegressiveUnet: wraps AutoregressiveUnet with BPE-prescribed gating.

The left-shift strategy places gate=1 at the first byte of each BPE token
(after the first), encoding token boundaries into prescribed_down_gate_samples.

Example:
    EvaByte:  <bos> h e l l o   w o r l d   t o k e n i z a t i o n <eos>
    BPE:      [BOS] [Hello   ] [_world    ] [_token    ] [ization        ] [EOS]
    gates:    1     0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0     1

Note: the model's gate_first_and_last_tokens() always forces gate=1 at
positions 0 and -1, so BOS/EOS boundaries are handled automatically.
"""

import os

import torch
from torch import nn
from tokenizers import Tokenizer

DEFAULT_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "trained_tokenizer")


def char_offset_to_byte_offset(text: str, char_pos: int) -> int:
    """Convert a character-level offset to a UTF-8 byte-level offset."""
    return len(text[:char_pos].encode("utf-8"))


class BPEAutoRegressiveUnet(nn.Module):
    """
    Wraps AutoregressiveUnet, converting text → prescribed_down_gate_samples
    using a trained BPE tokenizer and the left-shift gate strategy.

    Usage:
        wrapper = BPEAutoRegressiveUnet(model, tokenizer_path="bpe_tokenizer/trained_tokenizer")

        # Option A: pass texts alongside input_ids in the forward call
        out = wrapper(input_ids, texts=batch_texts)

        # Option B: pre-compute gates (e.g. in a collate_fn)
        gates = wrapper.texts_to_gate_tensors(batch_texts, seq_len, device)
        out = wrapper(input_ids, prescribed_down_gate_samples=gates)
    """

    def __init__(self, model, tokenizer_path: str = DEFAULT_TOKENIZER_PATH):
        super().__init__()
        self.model = model
        self.bpe_tokenizer = Tokenizer.from_file(
            os.path.join(tokenizer_path, "tokenizer.json")
        )

    def __getattr__(self, name):
        # Delegate any attribute not found on the wrapper to the wrapped model.
        # nn.Module stores its own attrs in __dict__ before this is called,
        # so self.model and self.bpe_tokenizer are always found directly.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def texts_to_gate_tensors(
        self, texts: list, seq_len: int, device
    ) -> torch.Tensor:
        """
        Batch-encode texts → prescribed_down_gate_samples of shape (B, seq_len).

        Gate=1 fires at the first byte of each BPE token after the first,
        relative to the EvaByte sequence (which has BOS at position 0).
        """
        encodings = self.bpe_tokenizer.encode_batch(texts)
        gates = torch.zeros(len(texts), seq_len, dtype=torch.long, device=device)

        for b, (text, enc) in enumerate(zip(texts, encodings)):
            # enc.offsets[0]  = BOS special token → (0, 0)
            # enc.offsets[-1] = EOS special token → (len(text), len(text))
            # Iterate real BPE tokens: indices 1 .. len(offsets)-2
            for i in range(1, len(enc.offsets) - 1):
                char_start = enc.offsets[i][0]
                byte_start = char_offset_to_byte_offset(text, char_start)
                pos = 1 + byte_start  # +1 to skip BOS at position 0
                if 0 < pos < seq_len - 1:
                    gates[b, pos] = 1

        # model's gate_first_and_last_tokens() already forces 0 and -1,
        # but set them here for correctness when used outside the model.
        gates[:, 0] = 1
        gates[:, -1] = 1
        return gates

    def forward(
        self,
        input_ids: torch.Tensor,
        texts=None,
        prescribed_down_gate_samples=None,
        **kwargs,
    ):
        """
        If texts is provided, compute prescribed_down_gate_samples from BPE
        boundaries and pass them to the underlying model. Otherwise, any
        explicitly provided prescribed_down_gate_samples (or None) is used.
        """
        if texts is not None:
            seq_len = input_ids.shape[1]
            prescribed_down_gate_samples = self.texts_to_gate_tensors(
                texts, seq_len, input_ids.device
            )
        return self.model(
            input_ids,
            prescribed_down_gate_samples=prescribed_down_gate_samples,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        size: str,
        architecture: str,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    ) -> "BPEAutoRegressiveUnet":
        """
        Convenience constructor: load model config and instantiate both
        AutoregressiveUnet and the BPE wrapper in one call.
        Compatible with the existing config_loader infrastructure.
        """
        from training_random_base_model.config_loader import load_model_config
        from model.model import AutoregressiveUnet

        model_kwargs = load_model_config(size, architecture)
        model = AutoregressiveUnet(**model_kwargs)
        return cls(model, tokenizer_path=tokenizer_path)
