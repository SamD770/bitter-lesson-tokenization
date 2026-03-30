I would like to train a BPE tokenizer on fineweb train until it compresses at a rate of roughly 0.2 bytes per token. Two important caveats:
- Separation with whitespace should be enforced as in SoTA tokenizers
- there should be beginning-of-sequence and end-of-sequence tokens

This bpe tokenizer should then be able to be used with the AutoregressiveUnet class in model.model by:

1. Predicting the token boundaries
2. Feeding in the predicted token boundaries into the forward pass using prescribed_down_gate_samples. This should use a single left-shift strategy, for example if the tokenizer splits " h e l l o _ w o r l d _ t o k e n i z a t i o n " as "Hello" "_world" "_token" "ization" then the prescribed_down_gate_samples should look like:

<bos> h e l l o _ w o r l d _ t o k e n i z a t i o n <eos>
1     0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1

Note that AutoregressiveUnet class already handles putting token boundaries at the first and last tokens. I _think_ the best way to do this would be to wrap the AutoregressiveUnet class, but I am happy to defer. Ideally, this would be compatible with the current configuration infrastructure setup. 