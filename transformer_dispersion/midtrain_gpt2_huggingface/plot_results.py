import argparse
import os
import numpy as np
from glob import glob
import json
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy

results_dict = {
    'dispersion': [],
    'dispersion_coeff': [],
    'dispersion_loc': [],
    'metrics': [],
}

empty_metrics_dict = {
    'step': [],
    'paloma_wikitext_103\nword_perplexity,none': {'mean': [], 'std': []},
    'hellaswag\nacc_norm,none': {'mean': [], 'std': []},
    'lambada_openai\nacc,none': {'mean': [], 'std': []},
    # 'lambada_standard\nacc,none': {'mean': [], 'std': []},
    # 'piqa\nacc,none': {'mean': [], 'std': []},
    'truthfulqa_mc2\nacc,none': {'mean': [], 'std': []},
    'winogrande\nacc,none': {'mean': [], 'std': []},
    'arc_challenge\nacc,none': {'mean': [], 'std': []},
    # 'gsm8k\nexact_match,flexible-extract': {'mean': [], 'std': []},
    'medmcqa\nacc,none': {'mean': [], 'std': []},
    'mmlu\nacc,none': {'mean': [], 'std': []},
}

def sort_by_step(steps, means, stds):
    order = np.argsort(np.array(steps))
    return np.array(steps)[order], np.array(means)[order], np.array(stds)[order]

def run_label(dispersion, coeff, loc):
    if str(dispersion) == 'None':
        return 'None'
    return f'{dispersion}-{coeff}-{loc}'

def extract_coeff_from_label(s):
    if s == 'Default loss':
        return 0
    elif s == 'No mid-training':
        return 'N/A'
    else:
        return float(s.split('-')[1].split('-')[0])

def _num_coeff(x):
    try: return float(x)
    except: return np.inf

def compute_best_steps(results_dict, selection_metrics):
    best_idx_per_run = {}
    sorted_cache = {}

    for ridx in range(len(results_dict['metrics'])):
        steps = np.asarray(results_dict['metrics'][ridx]['step'], dtype=int)
        if steps.size == 0:
            best_idx_per_run[ridx] = None
            continue
        order = np.argsort(steps)
        steps_sorted = steps[order]
        scores = np.full_like(steps_sorted, fill_value=np.nan, dtype=float)

        per_metric_vals = {}
        for m in selection_metrics:
            means = np.asarray(results_dict['metrics'][ridx][m]['mean'], dtype=float)
            if means.size == 0:
                continue
            means = means[order]
            per_metric_vals[m] = means

        for j in range(len(steps_sorted)):
            vals = []
            for m in selection_metrics:
                if m in per_metric_vals and np.isfinite(per_metric_vals[m][j]):
                    vals.append(per_metric_vals[m][j])
            if len(vals) > 0:
                scores[j] = float(np.mean(vals))

        if np.all(~np.isfinite(scores)):
            best_idx_per_run[ridx] = None
        else:
            best_idx_per_run[ridx] = int(np.nanargmax(scores))

        sorted_cache[(ridx, '__steps__')] = steps_sorted
        for m in results_dict['metrics'][ridx].keys():
            if m == 'step': continue
            means = np.asarray(results_dict['metrics'][ridx][m]['mean'], dtype=float)
            stds  = np.asarray(results_dict['metrics'][ridx][m]['std'], dtype=float)
            if means.size == 0:
                sorted_cache[(ridx, m)] = (steps_sorted, means, stds)
            else:
                sorted_cache[(ridx, m)] = sort_by_step(steps, means, stds)

    return best_idx_per_run, sorted_cache

def value_at_idx_pct(results_dict, sorted_cache, ridx, metric_name, step_idx, initial=False):
    steps_sorted = sorted_cache[(ridx, '__steps__')]
    means = np.asarray(results_dict['metrics'][ridx][metric_name]['mean'], dtype=float)
    if means.size == 0 or steps_sorted.size == 0:
        return np.nan
    _, means_sorted, _ = sorted_cache[(ridx, metric_name)]
    if means_sorted.size == 0:
        return np.nan
    idx = 0 if initial else step_idx
    if idx is None or idx >= means_sorted.size:
        return np.nan
    return float(means_sorted[idx]) * 100.0

def find_metric_ylims_by_best_step(results_dict, all_metric_names, baseline_idx, best_idx_per_run, sorted_cache):
    metric_ylim_bars = {m: [np.inf, -np.inf] for m in all_metric_names}
    for m in all_metric_names:
        cand = []
        # No mid-training initial
        steps_b = sorted_cache[(baseline_idx, '__steps__')]
        _, means_b, _ = sorted_cache[(baseline_idx, m)]
        if means_b.size > 0:
            cand.append(float(means_b[0]))
        # Default loss best overall
        bidx = best_idx_per_run[baseline_idx]
        if bidx is not None and means_b.size > bidx:
            cand.append(float(means_b[bidx]))
        # Other runs at their best overall
        for ridx in range(len(results_dict['metrics'])):
            if ridx == baseline_idx: continue
            _, means_r, _ = sorted_cache[(ridx, m)]
            j = best_idx_per_run[ridx]
            if means_r.size > 0 and j is not None and j < means_r.size and np.isfinite(means_r[j]):
                cand.append(float(means_r[j]))
        if cand:
            vmin = float(np.nanmin(cand)); vmax = float(np.nanmax(cand))
            metric_ylim_bars[m][0] = vmin; metric_ylim_bars[m][1] = vmax

    for m, (lo, hi) in metric_ylim_bars.items():
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        if hi == lo:
            eps = 1e-6 if lo == 0 else 0.01 * abs(lo)
            lo, hi = lo - eps, hi + eps
        pad = 0.05 * (hi - lo)
        metric_ylim_bars[m] = (lo - pad, hi + pad)

    return metric_ylim_bars

def render_latex_table_simple(
    results_dict,
    metric_names,
    baseline_idx,
    rows_by_dispersion,
    row_order,
    model_name,
    lora_suffix,
    decimals=1,
    out_path=None,
    best_idx_per_run=None,
    sorted_cache=None,
):
    rows = []
    rows.append({"method": "Pretrained (no mid-training)", "disp": "None", "coeff": "N/A", "loc": "-", "src": ("baseline","initial")})
    rows.append({"method": "Default loss", "disp": "None", "coeff": "0.0", "loc": "-", "src": ("baseline","best")})
    for disp in row_order:
        idxs = sorted(rows_by_dispersion[disp], key=lambda i: _num_coeff(results_dict['dispersion_coeff'][i]))
        for i in idxs:
            rows.append({
                "method": results_dict['dispersion'][i],
                "disp":   results_dict['dispersion'][i],
                "coeff":  results_dict['dispersion_coeff'][i],
                "loc":    results_dict['dispersion_loc'][i],
                "idx":    i,
                "src":    ("run","best"),
            })

    ref = {}
    for m in metric_names:
        val = value_at_idx_pct(results_dict, sorted_cache, baseline_idx, m, step_idx=None, initial=True)
        ref[m] = val

    align = "l c c " + " ".join(["c"]*len(metric_names)) + " c"
    headers = [m.replace("\n"," ").replace(",", " ") for m in metric_names]
    headers.append("Average")

    lines = []
    lines.append(r"\begin{tabular}{"+align+r"}")
    lines.append(r"\toprule")
    lines.append("Method & Coeff & Loc & " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        left = f"{row['method']} & {row['coeff']} & {row['loc']}"
        cells = []
        vals_for_avg = []
        bases_for_avg = []

        for m in metric_names:
            if row['src'] == ("baseline","initial"):
                val = ref[m]
                base = ref[m]
                cell = f"{val:.{decimals}f}" if np.isfinite(val) else "N/A"
            elif row['src'] == ("baseline","best"):
                bidx = best_idx_per_run[baseline_idx]
                val = value_at_idx_pct(results_dict, sorted_cache, baseline_idx, m, step_idx=bidx, initial=False)
                base = ref[m]
                cell = f"{val:.{decimals}f}"
                if np.isfinite(base):
                    diff = val - base
                    sgn  = "+" if diff >= 0 else ""
                    color = "forestgreen" if diff >= 0 else "crimson"
                    cell += f" \\textcolor{{{color}}}{{({sgn}{diff:.{decimals}f})}}"
            else:
                ridx = row['idx']
                j = best_idx_per_run[ridx]
                val = value_at_idx_pct(results_dict, sorted_cache, ridx, m, step_idx=j, initial=False)
                base = ref[m]
                cell = f"{val:.{decimals}f}"
                if np.isfinite(base):
                    diff = val - base
                    sgn  = "+" if diff >= 0 else ""
                    color = "forestgreen" if diff >= 0 else "crimson"
                    cell += f" \\textcolor{{{color}}}{{({sgn}{diff:.{decimals}f})}}"

            cells.append(cell)
            if np.isfinite(val):
                vals_for_avg.append(val)
            if np.isfinite(base):
                bases_for_avg.append(base)

        if vals_for_avg:
            avg_val = float(np.mean(vals_for_avg))
            avg_base = float(np.mean(bases_for_avg)) if bases_for_avg else np.nan
            if row['src'] == ("baseline","initial") or not np.isfinite(avg_base):
                avg_cell = f"{avg_val:.{decimals}f}"
            else:
                diff = avg_val - avg_base
                sgn  = "+" if diff >= 0 else ""
                color = "forestgreen" if diff >= 0 else "crimson"
                avg_cell = f"{avg_val:.{decimals}f} \\textcolor{{{color}}}{{({sgn}{diff:.{decimals}f})}}"
        else:
            avg_cell = "N/A"

        cells.append(avg_cell)
        lines.append(left + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_tex = "\n".join(lines)
    if out_path is None:
        os.makedirs("./tables", exist_ok=True)
        out_path = f"./tables/results_table_{model_name}{lora_suffix}.tex"
    with open(out_path, "w") as f:
        f.write(table_tex)

    print("\n===== LaTeX table (copy into Overleaf) =====\n")
    print(table_tex)
    print(f"\n[Saved LaTeX table to: {out_path}]\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Mid-train GPT-2 with a token budget.")
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) instead of full fine-tuning")
    ap.add_argument("--dataset_name", type=str, default="Salesforce/wikitext",
                    help="Hugging Face dataset id.")
    args = ap.parse_args()
    lora_suffix = "_lora" if args.lora else ""

    result_folder = './results/'
    figure_lines_save_path = f'./figures/results_lines_{args.model_name}{lora_suffix}.png'
    figure_bars_save_path = f'./figures/results_bars_{args.model_name}{lora_suffix}.png'

    os.makedirs(os.path.dirname(figure_lines_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_bars_save_path), exist_ok=True)
    run_folder_list = sorted(glob(os.path.join(result_folder, f'midtrain_{args.model_name}{lora_suffix}_{"-".join(args.dataset_name.split("/"))}_*')))

    for run_folder in run_folder_list:
        dispersion = run_folder.split('disp-')[1].split('-')[0]
        dispersion_coeff = run_folder.split(f'{dispersion}-')[1].split('-')[0]
        dispersion_loc = run_folder.split(f'{dispersion_coeff}-')[1].split('_')[0]

        if float(dispersion_coeff) > 1:
            continue

        results_dict['dispersion'].append(dispersion)
        results_dict['dispersion_coeff'].append(dispersion_coeff)
        results_dict['dispersion_loc'].append(dispersion_loc)
        results_dict['metrics'].append(deepcopy(empty_metrics_dict))

        eval_json_list = glob(os.path.join(run_folder, 'lm_eval_*.json'))

        for eval_json in eval_json_list:
            with open(eval_json, "r") as f:
                data_json = json.load(f)

            results_dict['metrics'][-1]['step'].append(int(eval_json.split('_')[-1].replace('.json', '')))
            for metric in empty_metrics_dict.keys():
                if metric != 'step':
                    metric_dataset = metric.split('\n')[0]
                    metric_measure = metric.split('\n')[1]
                    results_dict['metrics'][-1][metric]['mean'].append(
                        float(data_json['results'][metric_dataset][metric_measure]))
                    std = data_json['results'][metric_dataset][metric_measure.replace(',', '_stderr,')]
                    if std == 'N/A':
                        results_dict['metrics'][-1][metric]['std'].append(np.nan)
                    else:
                        results_dict['metrics'][-1][metric]['std'].append(float(std))

    all_metric_names = [k for k in results_dict['metrics'][0].keys() if k != 'step']

    baseline_idx = None
    for i, d in enumerate(results_dict['dispersion']):
        if d.lower() == "none":
            baseline_idx = i
            break
    if baseline_idx is None:
        raise RuntimeError("No baseline run found with dispersion == 'None'.")

    rows_by_dispersion = {}
    for i, d in enumerate(results_dict['dispersion']):
        if i == baseline_idx:
            continue
        rows_by_dispersion.setdefault(d, []).append(i)

    row_order = [d for d in ["Covariance", "Hinge", "InfoNCE_l2", "InfoNCE_cosine"] if d in rows_by_dispersion]

    selection_metrics = [m for m in all_metric_names if 'perplexity' not in m.lower()]
    best_idx_per_run, sorted_cache = compute_best_steps(results_dict, selection_metrics)

    metric_ylim_bars = find_metric_ylims_by_best_step(
        results_dict=results_dict,
        all_metric_names=all_metric_names,
        baseline_idx=baseline_idx,
        best_idx_per_run=best_idx_per_run,
        sorted_cache=sorted_cache,
    )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    fig_lines = plt.figure(figsize=(4*len(all_metric_names), 3*len(row_order)))
    fig_bars = plt.figure(figsize=(4*len(all_metric_names), 3*len(row_order)))

    for row_idx, row_disp in enumerate(row_order):
        run_indices = rows_by_dispersion[row_disp]
        cmap = cm.Reds

        for metric_idx, metric_name in enumerate(all_metric_names):
            ax_lines = fig_lines.add_subplot(len(row_order), len(all_metric_names), row_idx * len(all_metric_names) + metric_idx + 1)
            ax_lines.spines["top"].set_visible(False)
            ax_lines.spines["right"].set_visible(False)

            ax_bars = fig_bars.add_subplot(len(row_order), len(all_metric_names), row_idx * len(all_metric_names) + metric_idx + 1)
            ax_bars.spines["top"].set_visible(False)
            ax_bars.spines["right"].set_visible(False)
            bars_labels, bars_heights, bars_colors = [], [], []

            steps_b = sorted_cache[(baseline_idx, '__steps__')]
            _, means_b, _ = sorted_cache[(baseline_idx, metric_name)]
            if means_b.size > 0:
                bars_labels.append('No mid-training')
                bars_heights.append(float(means_b[0]))
                bars_colors.append('lightgray')

            b_best = best_idx_per_run[baseline_idx]
            if b_best is not None and means_b.size > b_best:
                bars_labels.append('Default loss')
                bars_heights.append(float(means_b[b_best]))
                bars_colors.append('gray')
                ax_lines.plot(steps_b, means_b, linestyle='--', linewidth=2, label='Default loss', color='black', alpha=0.5)

            for idx in run_indices:
                steps_r = sorted_cache[(idx, '__steps__')]
                _, means_r, _ = sorted_cache[(idx, metric_name)]
                j = best_idx_per_run[idx]
                coeff_logscaled = (np.log10(float(results_dict['dispersion_coeff'][idx])) + 4) / 7
                ax_lines.plot(steps_r, means_r, linewidth=2,
                              label=run_label(results_dict['dispersion'][idx],
                                              results_dict['dispersion_coeff'][idx],
                                              results_dict['dispersion_loc'][idx]),
                              color=cmap(coeff_logscaled))
                if means_r.size > 0 and j is not None and j < means_r.size:
                    bars_labels.append(run_label(results_dict['dispersion'][idx],
                                                 results_dict['dispersion_coeff'][idx],
                                                 results_dict['dispersion_loc'][idx]))
                    bars_heights.append(float(means_r[j]))
                    bars_colors.append(cmap(coeff_logscaled))

            ax_lines.set_xlabel("Step", fontsize=12)
            ax_lines.set_ylabel(metric_name, fontsize=12)

            bars = ax_bars.bar(np.arange(len(bars_labels)), bars_heights, color=bars_colors, alpha=0.8, label=bars_labels)
            if len(bars_heights) >= 2:
                ax_bars.axhline(y=bars_heights[1], linestyle='--', linewidth=2, color=bars_colors[1], alpha=0.8)
            ax_bars.set_xticks(np.arange(len(bars_labels)))
            ax_bars.set_xticklabels([extract_coeff_from_label(label) for label in bars_labels], rotation=0, ha='center', fontsize=9)
            ax_bars.set_ylabel(metric_name, fontsize=12)
            ax_bars.set_xlabel('Dispersion Coefficient', fontsize=12)
            ax_bars.set_ylim(metric_ylim_bars[metric_name])

            ax_bars.bar_label(
                bars,
                labels=[f"{v:.3f}" for v in bars_heights],
                rotation=90,
                padding=5,
                fontsize=9,
            )

            if metric_idx == 0:
                ax_lines.legend(fontsize=10, frameon=False, loc='upper right')
                legend_bars = ax_bars.legend(fontsize=12, frameon=False, loc='center left')
                ax_bars.cla()
                ax_bars.axis("off")
                ax_bars.add_artist(legend_bars)

    fig_lines.tight_layout(pad=2)
    fig_lines.savefig(figure_lines_save_path, dpi=300)

    fig_bars.tight_layout(pad=2)
    fig_bars.savefig(figure_bars_save_path, dpi=300)

    metrics_for_table = [m for m in all_metric_names if 'perplexity' not in m.lower()]
    render_latex_table_simple(
        results_dict=results_dict,
        metric_names=metrics_for_table,
        baseline_idx=baseline_idx,
        rows_by_dispersion=rows_by_dispersion,
        row_order=row_order,
        model_name=args.model_name,
        lora_suffix=lora_suffix,
        decimals=1,
        best_idx_per_run=best_idx_per_run,
        sorted_cache=sorted_cache,
    )
