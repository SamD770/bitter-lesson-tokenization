import os
import torch
from transformers import AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import HybridCache, Cache

from clean_code.flexible_bitter_llm import IndependentWrapperGater, FlexibleBitterLLM, FlexibleCache, DistributeTokenUpsampler


@torch.no_grad()
def generate_inefficient(model, starter_tokens, n_tokens, temperature=1.0, early_exit=False):
    # Don't use a kv cache and instead repeatedly pass expanding sequences to the model.
    token_ids = starter_tokens

    if early_exit:
        out_key = "early_logits"
    else:
        out_key = "logits"

    for _ in range(n_tokens):
        # Test: What if we make the model gate at every token? Result: no it doesn't screw up the model as much as the generate.
        # prescribed_gate_samples = torch.ones_like(token_ids, device=token_ids.device)

        out = model(
            token_ids,
            use_cache=False
        )

        continue_token_logits = out[out_key][:, -1, :]
        continue_token_logits = continue_token_logits / temperature
        continue_token_probs = torch.softmax(continue_token_logits, dim=-1)
        continue_token_id = torch.multinomial(continue_token_probs, num_samples=1)

        token_ids = torch.cat([token_ids, continue_token_id], dim=1)

    return token_ids

@torch.no_grad()
def generate(model, starter_tokens, n_tokens, temperature=1.0, early_exit=False):
    if early_exit:
        out_key = "early_logits"
    else:
        out_key = "logits"

    config = model.byte_layer_config

    past_key_value = FlexibleCache(
        config,
        max_batch_size=starter_tokens.shape[0],
        max_cache_len=1024,
        dtype=torch.float32,
        device=starter_tokens.device,
    )

    for l, l_key_cache in enumerate(past_key_value.key_cache):
        print(f"{l=} {l_key_cache.shape=} {(l_key_cache != 0).sum().item()=}")


    print(f"{past_key_value=}")


    out = model(
        starter_tokens, 
        past_key_value=past_key_value,
        use_cache=True
    )

    continue_token_ids = []

    for t in range(n_tokens):
        past_key_value = out["past_key_value"]
        continue_token_logits = out[out_key][:, -1, :]
        # Apply temperature scaling to logits
        continue_token_logits = continue_token_logits / temperature
        continue_token_probs = torch.softmax(continue_token_logits, dim=-1)
        continue_token_id = torch.multinomial(continue_token_probs, num_samples=1)

        continue_token_ids.append(continue_token_id)
        
        out = model(
            continue_token_id,
            past_key_value=past_key_value,
            use_cache=True
        )

    continue_token_ids = torch.cat(continue_token_ids, dim=1)
    return continue_token_ids


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float16
    byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")

    saved_model_file_name = f"training_random_base_model/random_select_early_output_base_model_42.pt"
    my_model = torch.load(os.path.join(net_scratch_dir, saved_model_file_name), weights_only=False).to(device)
    my_model.down_layer_gate = IndependentWrapperGater(my_model.down_layer_gate)
    my_model.upsampler = DistributeTokenUpsampler()

    # my_model = FlexibleBitterLLM(
    #     vocab_size=byte5_tokenizer.vocab_size, 
    #     embedding_dim=512, 
    #     num_heads=8, 
    #     downsample_rate=0.25, 
    #     sliding_window=64,
    #     flash_attn=False
    # ).to(device="cuda")
    
    my_model.byte_layer_config.num_hidden_layers = 10
    my_model.byte_layer_config.n_down_layers = 2
    my_model.byte_layer_config.n_mid_layers = 6
    my_model.byte_layer_config.n_up_layers = 2

    print(f"{my_model.byte_layer_config._attn_implementation=}")

    my_starter_string = "The BBC reported that t"
    my_starter = byte5_tokenizer.encode(my_starter_string, return_tensors="pt", add_special_tokens=False).to(device)

    # TODO: fix the cache.

    print("Efficient:")
    continue_token_ids = generate(my_model, my_starter, 300, temperature=1.)
    continuation = byte5_tokenizer.decode(continue_token_ids[0]).replace("\n", "\\n")
    print(my_starter_string, continuation, sep="|")
    
    print("Efficient Early exit:")
    continue_token_ids_early_exit = generate(my_model, my_starter, 300, temperature=1., early_exit=True)
    continuation_early_exit = byte5_tokenizer.decode(continue_token_ids_early_exit[0]).replace("\n", "\\n")
    print(my_starter_string, continuation_early_exit, sep="|")

    print("Inefficient:")
    continue_token_ids_inefficient = generate_inefficient(my_model, my_starter, 300, temperature=1.)
    continuation_inefficient = byte5_tokenizer.decode(continue_token_ids_inefficient[0]).replace("\n", "\\n")
    print(continuation_inefficient)

    print("Early exit:")
    continue_token_ids_early_exit = generate_inefficient(my_model, my_starter, 300, temperature=1., early_exit=True)
    continuation_early_exit = byte5_tokenizer.decode(continue_token_ids_early_exit[0]).replace("\n", "\\n")
    print(continuation_early_exit)

