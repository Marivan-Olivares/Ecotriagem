"""Treina e avalia o classificador de intenções e o extrator de entidades."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.machine_learning.avaliacao import (AZUL, metricas, plot_matriz_confusao,
                                            plt, salvar_json)
from src.pln.entidades import ExtratorEntidades
from src.pln.faq import LIMIAR_FAQ, BuscaFAQ
from src.pln.intencoes import INTENCOES, criar_modelo

CANDIDATOS = [
    ("naive_bayes", False), ("naive_bayes", True),
    ("regressao_logistica", False), ("regressao_logistica", True),
    ("svm_linear", False), ("svm_linear", True),
]
NOMES_CURTOS = {"identificar_material": "identificar", "como_descartar": "descartar",
                "ponto_coleta": "ponto coleta", "preparo_descarte": "preparo",
                "reciclavel": "reciclável", "tempo_decomposicao": "decomposição",
                "duvida_geral": "dúvida geral", "saudacao_ajuda": "saudação",
                "fora_escopo": "fora escopo"}


def prf(acertos, previstos, esperados):
    p = acertos / previstos if previstos else 0.0
    r = acertos / esperados if esperados else 0.0
    return {"precisao": p, "revocacao": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "acertos": acertos, "previstos": previstos, "esperados": esperados}


def main():
    treino = pd.read_csv(config.DIR_PLN / "intencoes_treino.csv")
    teste = pd.read_csv(config.DIR_PLN / "teste_manual.csv").fillna("")
    skf = StratifiedKFold(5, shuffle=True, random_state=config.SEMENTE)
    resultados = {"n_treino": len(treino), "n_teste_manual": len(teste), "modelos": {}}

    melhor, melhor_f1 = None, -1
    for nome, chars in CANDIDATOS:
        rotulo = f"{nome}{'+char' if chars else ''}"
        modelo = criar_modelo(nome, chars)
        pred_cv = cross_val_predict(modelo, treino["texto"], treino["intencao"], cv=skf)
        m_cv = metricas(treino["intencao"], pred_cv, INTENCOES)
        modelo.fit(treino["texto"], treino["intencao"])
        pred_te = modelo.predict(teste["texto"])
        m_te = metricas(teste["intencao"], pred_te, INTENCOES)
        resultados["modelos"][rotulo] = {
            "cv5_acuracia": m_cv["acuracia"], "cv5_f1_macro": m_cv["f1_macro"],
            "teste_acuracia": m_te["acuracia"], "teste_f1_macro": m_te["f1_macro"]}
        print(f"{rotulo:28s} CV F1={m_cv['f1_macro']:.3f} | teste manual "
              f"acc={m_te['acuracia']:.3f} F1={m_te['f1_macro']:.3f}")
        if m_cv["f1_macro"] > melhor_f1:
            melhor, melhor_f1 = (rotulo, modelo, pred_te, m_te), m_cv["f1_macro"]

    rotulo, modelo, pred_te, m_te = melhor
    print(f"modelo selecionado (maior F1 na validação cruzada): {rotulo}")
    joblib.dump(modelo, config.ARQ_MODELO_INTENCAO)
    resultados["selecionado"] = rotulo
    resultados["teste_selecionado"] = m_te
    proba = modelo.predict_proba(teste["texto"]).max(axis=1)
    resultados["confianca_media_acertos"] = float(proba[pred_te == teste["intencao"]].mean())
    resultados["confianca_media_erros"] = float(proba[pred_te != teste["intencao"]].mean())
    resultados["erros_teste"] = [
        {"texto": t, "esperado": e, "previsto": p, "confianca": float(c)}
        for t, e, p, c in zip(teste["texto"], teste["intencao"], pred_te, proba) if e != p]

    plot_matriz_confusao(teste["intencao"], pred_te, INTENCOES,
                         [NOMES_CURTOS[i] for i in INTENCOES],
                         config.DIR_GRAFICOS / "pln_matriz_confusao.png")

    nomes = list(resultados["modelos"])
    cv = [resultados["modelos"][n]["cv5_f1_macro"] for n in nomes]
    te = [resultados["modelos"][n]["teste_f1_macro"] for n in nomes]
    x = np.arange(len(nomes))
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar(x - 0.2, cv, 0.4, label="Validação cruzada (treino gerado)", color="#9dbbd0")
    ax.bar(x + 0.2, te, 0.4, label="Teste manual independente", color=AZUL)
    for xi, v in zip(x + 0.2, te):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [n.replace("regressao_logistica", "reg. log.").replace("naive_bayes", "NB")
                      .replace("svm_linear", "SVM") + (" ★" if n == rotulo else "")
                      for n in nomes], fontsize=8)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.15))
    fig.savefig(config.DIR_GRAFICOS / "pln_comparacao_modelos.png")
    plt.close(fig)

    ext = ExtratorEntidades()
    ac_m = prev_m = esp_m = ac_b = prev_b = esp_b = 0
    erros_ent = []
    for _, r in teste.iterrows():
        e = ext.extrair(r["texto"])
        if e.material:
            prev_m += 1
        if r["material"]:
            esp_m += 1
        if e.material and e.material == r["material"]:
            ac_m += 1
        if e.bairro:
            prev_b += 1
        if r["bairro"]:
            esp_b += 1
        if e.bairro and e.bairro == r["bairro"]:
            ac_b += 1
        if (e.material or "") != r["material"] or (e.bairro or "") != r["bairro"]:
            erros_ent.append({"texto": r["texto"], "material_esperado": r["material"],
                              "material_extraido": e.material or "",
                              "bairro_esperado": r["bairro"], "bairro_extraido": e.bairro or ""})
    resultados["entidades"] = {"material": prf(ac_m, prev_m, esp_m),
                               "bairro": prf(ac_b, prev_b, esp_b), "erros": erros_ent}
    print("entidades material:", {k: round(v, 3) for k, v in
                                   resultados['entidades']['material'].items()})
    print("entidades bairro:", {k: round(v, 3) for k, v in
                                 resultados['entidades']['bairro'].items()})

    faq = BuscaFAQ()
    tf = pd.read_csv(config.DIR_PLN / "teste_faq.csv").fillna("")
    acertos, detalhes = 0, []
    for _, r in tf.iterrows():
        resp, sim, perg = faq.responder(r["pergunta"])
        previsto = perg if resp else ""
        ok = previsto == r["faq_esperada"]
        acertos += ok
        detalhes.append({"pergunta": r["pergunta"], "esperada": r["faq_esperada"],
                         "recuperada": previsto, "similaridade": round(sim, 3), "acerto": bool(ok)})
    resultados["faq"] = {"acuracia_top1": acertos / len(tf), "n": len(tf),
                         "limiar": LIMIAR_FAQ, "detalhes": detalhes}
    print(f"FAQ top-1: {acertos}/{len(tf)}")
    salvar_json(resultados, config.DIR_METRICAS / "pln.json")


if __name__ == "__main__":
    main()
