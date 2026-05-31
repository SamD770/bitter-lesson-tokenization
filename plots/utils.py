import pandas as pd
import os
import json
import wandb
import html

def load_runs_from_wandb(run_names):

    wandb.login()
    api = wandb.Api()
    dfs = []
    configs = []

    for run_name in run_names:
        
        parquet_path = f"plots/datapoints/{run_name}.parquet"
        config_path= f"plots/datapoints/{run_name}.json"


        if not os.path.exists(parquet_path):
            my_run = api.run(f"{os.environ.get('WANDB_ENTITY')}/training_random_base_model/runs/{run_name}")

            history = my_run.scan_history()
            df = pd.DataFrame(history)
            df.to_parquet(parquet_path)

            config = my_run.config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

        else:
            df = pd.read_parquet(parquet_path)
            with open(config_path, "r") as f:
                config = json.load(f)

        configs.append(config)

        df['run_name'] = run_name
        # df['model_size'] = config['model_size']
        # df['GaterClass'] = config['GaterClass']

        # # For backward compatibility: we added these keys to the config later.
        # if 'UpsamplerClass' in config:
        #     df['UpsamplerClass'] = config['UpsamplerClass']

        # if "seed" in config:
        #     df['seed'] = config['seed']

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df, configs


def get_character_list(tokenizer, token_ids):
    """get a list of text characters from the list of EvaByte tokens. We handle multi-byte characters like so:
    [259, 244] -> ["", "é"]
    """
    # Build token display strings
    incomplete_marker=""
    undecoded_tokens = []
    token_displays = []
    
    for i, token_id in enumerate(token_ids):
        # Decode up to and including this token
        undecoded_tokens.append(token_id)
        decoded_with_new_token = tokenizer.decode(undecoded_tokens)
        
        # Check if new character(s) appeared
        if len(decoded_with_new_token) > 0:
            # This token completed one or more characters
            token_displays.append(decoded_with_new_token)
            undecoded_tokens = []
        else:
            # This token is part of an incomplete UTF-8 sequence
            token_displays.append(incomplete_marker)
        
    return token_displays


# Helper function: plot where the model puts high probability gates.
def gate_probs_html(txt, gate_probs, render_code=True):
    
    # Ensure gate_probs and input_text have the same length
    gate_probs = gate_probs[:len(txt)]
    
    # Create HTML with colored text based on probabilities
    colored_text = ""
    colorbar = ""
    
    # Create a colorbar showing the gradient
    for i in range(11):  # 0.0 to 1.0 in steps of 0.1
        prob = i / 10
        r = min(1.0, prob)
        b = max(0.0, 1.0 - prob)
        color = f"rgb({int(r*255)}, 0, {int(b*255)})"
        white = f"rgb(255, 255, 255)"
        colorbar += f'<span style="color:{white}; background-color:{color}; margin-right:2px; padding:0 5px;">{prob:.1f}</span>'
    
    # Add a legend for the colorbar
    colorbar_html = f'''
    <div style="margin-bottom:10px;">
        <div style="font-family:monospace; font-size:12px; margin-bottom:3px;">Token boundary probability: Low to High</div>
        <div style="font-family:monospace; font-size:14px;">{colorbar}</div>
    </div>
    '''
    
    # Process the text with colors
    for char, prob in zip(txt, gate_probs):
        # Use html.escape to properly handle all special characters including accented ones
        # import html
        char = html.escape(char)
        if break_newline:
            char = char.replace("\n", "\\n<br>")
            char = char.replace(" ", "&nbsp;")
        else:
            char = char.replace("\n", "\\n")
        char = char.replace("\t", "\\t")
        # 
        # Convert probability to color (blue->red)
        r = min(1.0, prob)  # Red increases with probability
        b = max(0.0, 1.0 - prob)  # Blue decreases with probability
        color = f"rgb({int(r*255)}, 0, {int(b*255)})"
        # Add the colored character to the output
        colored_text += f'<span style="color:{white}; background-color:{color};">{char}</span>'
    
    # Display the colorbar and colored text
    return f'''
    <meta charset="utf-8">
    <div>
        {colorbar_html}
        <div style="font-family:monospace; font-size:14px; word-wrap: break-word; overflow-wrap: break-word; white-space: pre-wrap; max-width: 100%;">{colored_text}</div>
    </div>
    '''