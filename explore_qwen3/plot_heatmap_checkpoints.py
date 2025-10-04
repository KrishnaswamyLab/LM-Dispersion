from typing import List
import os
import sys
from glob import glob
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel
import tempfile
import gc
import argparse

sys.path.append('../')
from prelim.utils.text_data import get_random_long_text


def normalize(x, p=2, axis=1, eps=1e-3):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def organize_embeddings(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    embeddings_by_layer = []
    for z in embeddings:
        z = z.squeeze(0).cpu().numpy()
        embeddings_by_layer.append(z)
    return embeddings_by_layer

def compute_cosine_similarities(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in embeddings:
        z = normalize(z, axis=1)
        cossim_matrix = np.matmul(z, z.T).clip(-1, 1)
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer

def build_hist_stack(cossim_matrix_by_layer: List[np.ndarray], step: int = 1, bins: int = 128):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
    layer_indices, hist_data = [], []
    for (layer_idx, cossim_matrix) in selected:
        cossim_arr = cossim_matrix.flatten()
        hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
        hist_data.append(hist)
        layer_indices.append(layer_idx)
    return np.array(hist_data), layer_indices

def parse_run_triplet(run_folder: str):
    dispersion = run_folder.split('disp-')[1].split('-')[0]
    dispersion_coeff = run_folder.split(f'{dispersion}-')[1].split('-')[0]
    dispersion_loc = run_folder.split(f'{dispersion_coeff}-')[1].split('_')[0]
    return dispersion, dispersion_coeff, dispersion_loc

def find_checkpoints(run_folder: str):
    ckpt_dirs = glob(os.path.join(run_folder, 'eval_ckpt_*_step*'))
    out = []
    for p in ckpt_dirs:
        name = os.path.basename(p)
        if 'step' in name:
            try:
                step = int(name.split('step')[-1])
                out.append((step, p))
            except:
                pass
    out.sort(key=lambda x: x[0])
    return out

def run_label(d, c, l):
    if str(d) == 'None':
        return 'None'
    return f'{d}-{c}-{l}'


def coeff_key(x):
    try:
        return float(x)
    except:
        return np.inf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embedding heatmap for a single model')
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name (e.g., Qwen/Qwen3-0.6B-Base)')
    args = parser.parse_args()
    
    model_name = args.model_name
    
    # Create safe filename from model name
    model_key = model_name.replace('/', '_')
    
    figures_dir = './figures'
    data_dir = './data'
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    figure_save_path = f'{figures_dir}/embedding_heatmap_{model_key}.png'
    data_save_path = f'{data_dir}/embedding_heatmap_{model_key}.npz'

    repetitions = 5
    max_length = 1024
    vmax = 10
    ckpt_stride = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Processing model: {model_name}")
    print(f"Using cache directory: {cache_dir}")
    print(f"Device: {device}")
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        print(f"  Model loaded successfully")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()
    print(f"  Computing embeddings ({repetitions} repetitions)...")

    cossim_matrix_by_layer = None
    for random_seed in range(repetitions):
        torch.manual_seed(random_seed)
        text = get_random_long_text('wikipedia',
                                    random_seed=random_seed,
                                    min_word_count=1024,
                                    max_word_count=1280)
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            output = model(**tokens, output_hidden_states=True)
            embeddings_by_layer = organize_embeddings(output.hidden_states)
            curr = compute_cosine_similarities(embeddings_by_layer)

        if cossim_matrix_by_layer is None:
            cossim_matrix_by_layer = [m[None, ...] for m in curr]
        else:
            for i in range(len(cossim_matrix_by_layer)):
                cossim_matrix_by_layer[i] = np.concatenate(
                    (cossim_matrix_by_layer[i], curr[i][None, ...]), axis=0
                )

    for i in range(len(cossim_matrix_by_layer)):
        cossim_matrix_by_layer[i] = cossim_matrix_by_layer[i].mean(axis=0)

    hist_matrix, layer_indices = build_hist_stack(cossim_matrix_by_layer)
    
    if hist_matrix.size == 0:
        print("Warning: Empty histogram matrix")
        ax.axis('off')
    else:
        im = ax.imshow(
            hist_matrix,
            aspect='auto',
            origin='lower',
            cmap='Reds',
            extent=[-1, 1, 0, layer_indices[-1]],
            vmin=0,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label='Density')

    ax.set_title(f'{model_name}', fontsize=16)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_xlabel('Cosine Similarity', fontsize=14)

    del model, tokenizer, config
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    fig.tight_layout()
    fig.savefig(figure_save_path, dpi=300)
    
    # Save intermediate data
    np.savez(data_save_path, 
             hist_matrix=hist_matrix, 
             layer_indices=layer_indices,
             model_name=model_name)
    
    print(f'\nSaved figure to: {figure_save_path}')
    print(f'Saved data to: {data_save_path}')
    print('Done.')
