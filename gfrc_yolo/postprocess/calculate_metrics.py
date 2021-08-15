import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def conf_mat_raw(true, predicted, labels):
    mat_out = np.empty((len(labels), len(labels)))
    for i, row in enumerate(labels):
        preds_row = predicted[true == row]
        for j, col in enumerate(labels):
            mat_out[i, j] = np.sum(preds_row == col)
    mat_out = np.array(mat_out, dtype=np.int)
    return mat_out


def conf_mat_plot_heatmap_blankTN(cm, display_labels, title_in, heatmap_type='true'):
    if len(display_labels) == 2:
        fig, ax = plt.subplots(figsize=(6,4.5))
    else:
        fig, ax = plt.subplots(figsize=(12,9))
    n_classes = cm.shape[0]
    cmap = 'Greys'

    if heatmap_type == 'percent':
        sum_vals = np.sum(cm)
    elif heatmap_type == 'true':
        sum_vals = np.reshape(np.repeat(np.sum(cm, axis=1), n_classes), (n_classes, n_classes))
    elif heatmap_type == 'pred':
        sum_vals = np.reshape(np.tile(np.sum(cm, axis=0), n_classes), (n_classes, n_classes))
        print(sum_vals)

    color_mapping = np.array(np.multiply(np.divide(cm, sum_vals), 255), np.uint8)

    for i in range(n_classes):
        for j in range(n_classes):
            if i == 0 and j == 0:
                text_cm = ""
            else:
                text_cm = format(cm[i, j], ',')
            txt_color = [1, 1, 1] if color_mapping[i, j] > 100 else [0, 0, 0]
            ax.text(j, i, text_cm, ha="center", va="center", color=txt_color, fontsize=18)
            ax.axhline(i - .5, color='black', linewidth=1.0)
            ax.axvline(j - .5, color='black', linewidth=1.0)

    ax.matshow(color_mapping, cmap=cmap)

    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(display_labels, fontsize=16)
    ax.set_yticklabels(display_labels, fontsize=16)
    ax.set_title(title_in, fontsize=16)
    ax.tick_params(bottom=True, labelbottom=True, top=False, labeltop=False)

    ax.set_ylim((n_classes - 0.5, -0.5))

    return ax



def conf_mat_plot_heatmap_blankTN_no_title(cm, display_labels, heatmap_type='true'):
    if len(display_labels) == 2:
        fig, ax = plt.subplots(figsize=(6,4.5))
    else:
        fig, ax = plt.subplots(figsize=(12,9))
    n_classes = cm.shape[0]
    cmap = 'Greys'

    if heatmap_type == 'percent':
        sum_vals = np.sum(cm)
    elif heatmap_type == 'true':
        sum_vals = np.reshape(np.repeat(np.sum(cm, axis=1), n_classes), (n_classes, n_classes))
    elif heatmap_type == 'pred':
        sum_vals = np.reshape(np.tile(np.sum(cm, axis=0), n_classes), (n_classes, n_classes))
        print(sum_vals)

    color_mapping = np.array(np.multiply(np.divide(cm, sum_vals), 255), np.uint8)

    for i in range(n_classes):
        for j in range(n_classes):
            if i == 0 and j == 0:
                text_cm = ""
            else:
                text_cm = format(cm[i, j], ',')
            txt_color = [1, 1, 1] if color_mapping[i, j] > 100 else [0, 0, 0]
            ax.text(j, i, text_cm, ha="center", va="center", color=txt_color, fontsize=18)
            ax.axhline(i - .5, color='black', linewidth=1.0)
            ax.axvline(j - .5, color='black', linewidth=1.0)

    ax.matshow(color_mapping, cmap=cmap)

    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(display_labels, fontsize=16)
    ax.set_yticklabels(display_labels, fontsize=16)
    ax.tick_params(bottom=True, labelbottom=True, top=False, labeltop=False)

    ax.set_ylim((n_classes - 0.5, -0.5))

    return ax


def save_conf_mat_plot(cm, labels, title, results_dir, prefix):
    n_class = len(labels)
    cm_all = np.reshape(np.array(cm, dtype=np.int), (n_class, n_class))
    cm_out = conf_mat_plot_heatmap_blankTN(cm_all, labels, title)
    out_path = prefix + '_confidence_matrix.png'
    results_dir = Path(results_dir)
    cm_out.get_figure().savefig(results_dir / out_path)


def save_conf_mat_plot_no_title(cm, labels, results_dir, prefix):
    n_class = len(labels)
    cm_all = np.reshape(np.array(cm, dtype=np.int), (n_class, n_class))
    cm_out = conf_mat_plot_heatmap_blankTN_no_title(cm_all, labels)
    out_path = prefix + '_confidence_matrix.png'
    results_dir = Path(results_dir)
    cm_out.get_figure().savefig(results_dir / out_path)