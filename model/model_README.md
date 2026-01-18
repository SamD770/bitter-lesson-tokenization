This is a research repository for developing a new method of dynamic tokenization based on score function estimators. It is designed to orchestrate the computations of the general class of dynamic tokenization methods, such that they can all be benchmarked in the same repo. The general structure of the forward pass is as follows:

We first pass the bytes into a local encoder to get a series of `down_layers`, which give byte-level embeddings:

```
x: (batch_size, sequence_length, embedding_dim) 
```

These byte embeddings are passed to the `Gater` which implements:

```
down_gate_logits : (batch_size, sequence_length) 
down_gate_probs: (batch_size, sequence_length)
down_gate_samples: (batch_size, sequence_length)
```

`down_gate_samples[b, l]` should be `1` if there will be a token boundary at the corresponding byte and `0` otherwise. `down_gate_probs`should be bounded in the interval `[0, 1]` and `down_gate_logits` has no restrictions. For score function methods, `down_gate_logits[b, l] = sigmoid(down_gate_logits[b, l])`. The `down_gate_samples` are used by the `Downsampler` to get the downsampled token-level sequence length:

```
x_downsampled: (batch_size, max_token_sequence_length, embedding_dim) 
```

This process also can produce `down_merge_dst: (batch_size, sequence_length) `: a tensor which points from a byte embedding to the token index which it was merged into.

TheThese token-level representations are then fed through the `mid_layers`. The `Upsampler` then combines all these previous values into a new set of byte-level embeddings, which are then fed through the `up_layers`.


Crucially, *managing the computation graph to make sure that the gradient is faithfully computed as described by each method is delegated to the `Gater`, `Upsampler` and `Downsampler`.*

Currently not implemented optimization:
- KV caching for inference
- sequence packing
- binning of sequence lengths (for `torch.compile` support)