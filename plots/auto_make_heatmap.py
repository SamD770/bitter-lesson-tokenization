
from IPython.display import HTML
from plots.utils import gate_probs_html, get_character_list
from model.model import text_to_tensor
import os
from training_random_base_model.model_from_checkpoint import load_model
import torch
from data_processing import split_fineweb, split_codeparrot
from transformers import AutoTokenizer


def auto_make_heatmap(checkpoint_directory, render_directory=None, dataset="fineweb"):

    if render_directory is None:
        # Extract the checkpoint name from the full path
        checkpoint_name = os.path.basename(checkpoint_directory)
        # Replace 'checkpoints' with 'renders' in the parent directory
        parent_dir = os.path.dirname(checkpoint_directory)
        render_parent = parent_dir.replace("checkpoints", "renders")
        render_directory = os.path.join(render_parent, checkpoint_name)

    model = load_model(checkpoint_directory)
    model.eval()
    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)

    if dataset == "fineweb":
        _, _, test_set = split_fineweb.get_splits()
        break_newline = False
    elif dataset == "codeparrot":
        _, _, test_set = split_codeparrot.get_splits()
        break_newline = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}.")

    model = model.to("cuda", dtype=torch.bfloat16)

    render_and_save_heatmaps(model, test_set, byte_tokenizer, render_directory, break_newline=break_newline)


def render_and_save_heatmaps(model, test_set, byte_tokenizer, render_directory, break_newline=False):
    for data_index in [1, 3, 5, 7]:
        text = test_set[data_index]["text"]
        tokens, _ = text_to_tensor(test_set[data_index], byte_tokenizer, 4096, "cuda")
        token_list = get_character_list(byte_tokenizer, tokens[0])

        with torch.no_grad():
            texts_arg = [text] if hasattr(model, "bpe_tokenizer") else None
            out = model(tokens, texts=texts_arg) if texts_arg else model(tokens)

        if not os.path.exists(render_directory):
            os.makedirs(render_directory)

        for property in ["probs", "samples"]:
            html_code = gate_probs_html(token_list, out[f"down_gate_{property}"][0], break_newline=break_newline)
            filename = f"{render_directory}/heatmaps_{property}_{data_index}.html"
            with open(filename, "w") as f:
                f.write(html_code)
            print(f"Heatmap saved to {filename}")