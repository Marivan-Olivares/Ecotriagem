"""Treina, seleciona e avalia os classificadores visuais."""
import sys
import time
from pathlib import Path

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.machine_learning.avaliacao import (AZUL, metricas, plot_matriz_confusao,
                                            plt, salvar_json)
from src.machine_learning.modelos import candidatos


AUROC_MIN, ECE_MAX = 0.75, 0.05
ACURACIA_ALVO_ALTA = 0.95
COBERTURA_MIN_ALTA = 0.50


def ece(conf, acerto, faixas=10):
    total = 0.0
    for lo in np.linspace(0, 1, faixas + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / faixas)
        if m.any():
            total += m.mean() * abs(acerto[m].mean() - conf[m].mean())
    return float(total)


def estimar_limiares(conf, acerto):
    """alta: menor τ com acurácia(conf>=τ) >= alvo e cobertura >= mínima;"""
    grade = np.round(np.arange(0.30, 0.99, 0.01), 2)
    alta = next((float(t) for t in grade if (conf >= t).mean() >= COBERTURA_MIN_ALTA
                 and acerto[conf >= t].mean() >= ACURACIA_ALVO_ALTA), 0.70)
    minima = 0.30
    for t in grade[grade < alta]:
        abaixo = conf < t
        if abaixo.sum() >= 5 and acerto[abaixo].mean() <= 0.5:
            minima = float(t)
    return {"alta": alta, "minima": minima,
            "cobertura_alta_val": float((conf >= alta).mean()),
            "acuracia_alta_val": float(acerto[conf >= alta].mean()),
            "fracao_abaixo_minima_val": float((conf < minima).mean()),
            "acuracia_abaixo_minima_val": float(acerto[conf < minima].mean())
            if (conf < minima).any() else None}


def carregar():
    d = np.load(config.DIR_CACHE / "caracteristicas.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def conjunto(d, carac, particao, aumento=False):
    m = d["particao"] == particao
    X, y, g = d[carac][m], d["classe"][m], d["grupo"][m]
    if aumento:
        X = np.vstack([X, d["cnn_espelhada"][m]])
        y, g = np.concatenate([y, y]), np.concatenate([g, g])
    return X, y, g


def main():
    d = carregar()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEMENTE)
    res, modelos_treinados = {}, {}
    for nome, (carac, aumento, pipe, grade) in candidatos().items():
        Xtr, ytr, gtr = conjunto(d, carac, "treino", aumento)
        Xva, yva, _ = conjunto(d, carac, "validacao")
        Xte, yte, _ = conjunto(d, carac, "teste")
        t0 = time.time()
        busca = GridSearchCV(clone(pipe), grade, scoring="f1_macro", cv=cv, n_jobs=2)
        busca.fit(Xtr, ytr, groups=gtr)
        t_treino = time.time() - t0
        modelo = busca.best_estimator_
        t0 = time.time()
        pva = modelo.predict(Xva)
        pte = modelo.predict(Xte)
        t_pred = (time.time() - t0) / (len(Xva) + len(Xte)) * 1000
        mva = metricas(yva, pva, config.CLASSES)
        mte = metricas(yte, pte, config.CLASSES)
        cva = modelo.predict_proba(Xva).max(1)
        acerto_va = pva == yva
        auroc_va = float(roc_auc_score(acerto_va, cva))
        ece_va = ece(cva, acerto_va)
        res[nome] = {"caracteristica": carac, "aumento": aumento,
                     "melhores_hiperparametros": busca.best_params_,
                     "cv5_f1_macro": float(busca.best_score_),
                     "val_acuracia": mva["acuracia"], "val_f1_macro": mva["f1_macro"],
                     "val_auroc_confianca": auroc_va, "val_ece": ece_va,
                     "confianca_informativa": auroc_va >= AUROC_MIN and ece_va <= ECE_MAX,
                     "teste_acuracia": mte["acuracia"], "teste_f1_macro": mte["f1_macro"],
                     "teste_por_classe": mte["por_classe"],
                     "tempo_treino_s": t_treino, "tempo_predicao_ms": t_pred}
        modelos_treinados[nome] = modelo
        print(f"{nome:34s} CV={busca.best_score_:.3f} val F1={mva['f1_macro']:.3f} "
              f"AUROC={auroc_va:.3f} ECE={ece_va:.3f} | "
              f"teste acc={mte['acuracia']:.3f} F1={mte['f1_macro']:.3f} "
              f"({t_treino:.0f}s) {busca.best_params_}")

    elegiveis = [n for n in res if res[n]["confianca_informativa"]] or list(res)
    melhor_so_f1 = max(res, key=lambda n: res[n]["val_f1_macro"])
    melhor = max(elegiveis, key=lambda n: res[n]["val_f1_macro"])
    print(f"\nmaior F1 na validação (sem restrição): {melhor_so_f1}")
    print(f"selecionado (F1 + confiança informativa/calibrada): {melhor}")
    carac = res[melhor]["caracteristica"]
    modelo = modelos_treinados[melhor]
    Xva, yva, _ = conjunto(d, carac, "validacao")
    Pva = modelo.predict_proba(Xva)
    limiares = estimar_limiares(Pva.max(1), np.array(modelo.classes_)[Pva.argmax(1)] == yva)
    config.ARQ_LIMIARES.write_text(json.dumps(limiares, indent=2), encoding="utf-8")
    print("limiares (validação):", limiares)
    joblib.dump({"modelo": modelo, "caracteristica": carac, "classes": list(modelo.classes_),
                 "nome": melhor}, config.ARQ_MODELO_VISAO)

    Xte, yte, _ = conjunto(d, carac, "teste")
    pte = modelo.predict(Xte)
    nomes = [config.CLASSES_PT[c] for c in config.CLASSES]
    plot_matriz_confusao(yte, pte, config.CLASSES, nomes,
                         config.DIR_GRAFICOS / "visao_matriz_confusao.png")
    melhor_cls = max([n for n in res if res[n]["caracteristica"] == "classica"],
                     key=lambda n: res[n]["val_f1_macro"])
    Xc, yc, _ = conjunto(d, "classica", "teste")
    plot_matriz_confusao(yc, modelos_treinados[melhor_cls].predict(Xc), config.CLASSES, nomes,
                         config.DIR_GRAFICOS / "visao_matriz_confusao_classico.png")

    ordem = list(res)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(ordem))
    ax.barh(y + 0.2, [res[n]["val_f1_macro"] for n in ordem], 0.4, color="#9dbbd0",
            label="Validação")
    ax.barh(y - 0.2, [res[n]["teste_f1_macro"] for n in ordem], 0.4, color=AZUL, label="Teste")
    for yi, n in zip(y, ordem):
        ax.text(res[n]["teste_f1_macro"] + 0.01, yi - 0.2, f"{res[n]['teste_f1_macro']:.3f}",
                va="center", fontsize=7)
    ax.set_yticks(y, [n + ("  ★" if n == melhor else "") for n in ordem], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-macro")
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.45, 1.14))
    fig.savefig(config.DIR_GRAFICOS / "visao_comparacao_modelos.png")
    plt.close(fig)

    salvar_json({"modelos": res, "selecionado": melhor, "maior_f1_sem_restricao": melhor_so_f1,
                 "criterio": {"auroc_min": AUROC_MIN, "ece_max": ECE_MAX},
                 "limiares": limiares,
                 "tempo_extracao_cnn_ms": float(d["tempo_cnn_ms"]),
                 "tempo_extracao_classica_ms": float(d["tempo_classica_ms"]),
                 "n": {p: int((d["particao"] == p).sum()) for p in ("treino", "validacao", "teste")}},
                config.DIR_METRICAS / "visao.json")


if __name__ == "__main__":
    main()
