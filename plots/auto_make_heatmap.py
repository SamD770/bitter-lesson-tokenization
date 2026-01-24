
from IPython.display import HTML
from plots.utils import gate_probs_html, get_character_list
from model.model import text_to_tensor
import os
from training_random_base_model.model_from_checkpoint import load_model
import torch
from data_processing import split_fineweb
from transformers import AutoTokenizer


def auto_make_heatmap(checkpoint_directory, render_directory):
    model = load_model(checkpoint_directory)
    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)

    _, _, test_set = split_fineweb.get_splits()
    model = model.to("cuda", dtype=torch.bfloat16)

    render_and_save_heatmaps(model, test_set, byte_tokenizer, render_directory)


def render_and_save_heatmaps(model, test_set, byte_tokenizer, render_directory):
    for data_index in [1, 3, 5, 7]:
        tokens, _ = text_to_tensor(test_set[data_index], byte_tokenizer, 4096, "cuda")
        token_list = get_character_list(byte_tokenizer, tokens[0])
        out = model(tokens)

        if not os.path.exists(render_directory):
            os.makedirs(render_directory)

        for property in ["probs", "samples"]:
            html_code = gate_probs_html(token_list, out[f"down_gate_{property}"][0])
            filename = f"{render_directory}/heatmaps_{property}_{data_index}.html"
            with open(filename, "w") as f:
                f.write(html_code)
            print(f"Heatmap saved to {filename}")