"""Métricas e gráficos de avaliação compartilhados pelos experimentos."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight"})

AZUL = "#2a6f97"
CORES = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644"]


def metricas(y_true, y_pred, rotulos) -> dict:
    rel = classification_report(y_true, y_pred, labels=rotulos, output_dict=True,
                                zero_division=0)
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=rotulos, average="macro",
                                   zero_division=0)),
        "f1_ponderado": float(f1_score(y_true, y_pred, labels=rotulos,
                                       average="weighted", zero_division=0)),
        "por_classe": {r: {k: float(v) for k, v in rel[r].items()} for r in rotulos},
    }


def salvar_json(obj, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def plot_matriz_confusao(y_true, y_pred, rotulos, nomes, caminho, titulo=""):
    cm = confusion_matrix(y_true, y_pred, labels=rotulos)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(0.62 * len(rotulos) + 1.6, 0.55 * len(rotulos) + 1.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.55 else "#222")
    ax.set_xticks(range(len(rotulos)), nomes, rotation=45, ha="right")
    ax.set_yticks(range(len(rotulos)), nomes)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    if titulo:
        ax.set_title(titulo)
    ax.spines[:].set_visible(False)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho)
    plt.close(fig)
    return cm
