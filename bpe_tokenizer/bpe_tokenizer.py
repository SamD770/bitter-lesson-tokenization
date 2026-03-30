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
        import time as _time
        _t0 = _time.perf_counter()

        # Truncate to seq_len bytes before tokenizing — texts can be arbitrarily long
        # but gate positions beyond seq_len are ignored. This avoids O(text_len * n_tokens)
        # work in char_offset_to_byte_offset for very long documents.
        texts_truncated = [t.encode("utf-8")[:seq_len].decode("utf-8", errors="ignore") for t in texts]
        _t1 = _time.perf_counter()

        encodings = self.bpe_tokenizer.encode_batch(texts_truncated)
        _t2 = _time.perf_counter()

        gates = torch.zeros(len(texts), seq_len, dtype=torch.long)  # CPU — filled with scalar writes, then moved to device
        _t3 = _time.perf_counter()

        for b, (text, enc) in enumerate(zip(texts_truncated, encodings)):
            # enc.offsets[0]  = BOS special token → (0, 0)
            # enc.offsets[-1] = EOS special token → (len(text), len(text))
            # Iterate real BPE tokens: indices 1 .. len(offsets)-2
            for i in range(1, len(enc.offsets) - 1):
                char_start = enc.offsets[i][0]
                byte_start = char_offset_to_byte_offset(text, char_start)
                pos = 1 + byte_start  # +1 to skip BOS at position 0
                if 0 < pos < seq_len - 1:
                    gates[b, pos] = 1
        _t4 = _time.perf_counter()

        # model's gate_first_and_last_tokens() already forces 0 and -1,
        # but set them here for correctness when used outside the model.
        gates[:, 0] = 1
        gates[:, -1] = 1
        result = gates.to(device)
        _t5 = _time.perf_counter()

        # print(f"[texts_to_gate_tensors] truncate={_t1-_t0:.3f}s  encode_batch={_t2-_t1:.3f}s  zeros={_t3-_t2:.3f}s  loop={_t4-_t3:.3f}s  to_device={_t5-_t4:.3f}s  total={_t5-_t0:.3f}s  batch={len(texts)}  n_tokens_ex={len(encodings[0].offsets) if encodings else 0}")
        return result

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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_directory: str,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    ) -> "BPEAutoRegressiveUnet":
        from training_random_base_model.config_loader import load_model_config_from_file
        from model.model import AutoregressiveUnet
        from safetensors.torch import load_file
        import os

        model_config = load_model_config_from_file(os.path.join(checkpoint_directory, "model_config.json"))
        inner_model = AutoregressiveUnet(**model_config)
        wrapper = cls(inner_model, tokenizer_path=tokenizer_path)
        model_state = load_file(os.path.join(checkpoint_directory, "model.safetensors"))
        wrapper.load_state_dict(model_state, strict=False)
        return wrapper
