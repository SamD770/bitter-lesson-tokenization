"""
We use the Evabyte tokenizer instead of the ByT5 tokenizer, as we want to use the BOS token as the first token.
See "The Curious Case of the bos_token" https://www.lesswrong.com/posts/tr3DrQiuyxkDpPqx2/the-curious-case-of-the-bos_token for why one might want to do this.
(TL;DR: the bos_token is important for the model to use it as a no-op in the attention mechanism).
"""

from transformers import AutoTokenizer

evabyte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

for tokenizer in [evabyte_tokenizer, byt5_tokenizer]:
    my_seqs = ["Hello, world!", "ABCDEF!"]
    print(f"{tokenizer.vocab_size=}")
    print(f"{len(tokenizer)=}")

    print(tokenizer(my_seqs, return_tensors="pt", padding=True))

    print(tokenizer.batch_decode(tokenizer(my_seqs, return_tensors="pt", padding=True)["input_ids"]))

    print("-"*100)