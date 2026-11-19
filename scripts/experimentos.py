"""Experimentos de avaliação do EcoTriagem (gera métricas e gráficos reais)."""
import itertools
import json
import random
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.decisao.base_regras import REGRAS
from src.decisao.busca import MapaColeta
from src.decisao.regras import MotorRegras
from src.machine_learning.avaliacao import AZUL, CORES, plt, salvar_json
from src.pln.entidades import materiais_em_rotulo
from src.pln.preprocessamento import normalizar
from src.visao_computacional import ocr
from src.visao_computacional import preprocessamento as pp
from src.visao_computacional.classificador import MAPA_MATERIAL
from src.visao_computacional.extrator_classico import extrair_lote
from src.visao_computacional.extrator_cnn import ExtratorMobileNet

G = config.DIR_GRAFICOS
ARQ_SAIDA = config.DIR_METRICAS / "experimentos.json"
NOMES = [config.CLASSES_PT[c] for c in config.CLASSES]


def carregar_cache():
    d = np.load(config.DIR_CACHE / "caracteristicas.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def pacote_visual():
    return joblib.load(config.ARQ_MODELO_VISAO)


def kb():
    return {k: v for k, v in json.load(open(config.DIR_CONHECIMENTO / "materiais.json",
                                            encoding="utf-8")).items() if not k.startswith("_")}


def e1_confianca(d, pac):
    modelo = pac["modelo"]
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for particao, cor in (("validacao", "#9dbbd0"), ("teste", AZUL)):
        m = d["particao"] == particao
        P = modelo.predict_proba(d[pac["caracteristica"]][m])
        y = d["classe"][m]
        pred = np.array(modelo.classes_)[P.argmax(1)]
        conf = P.max(1)
        acerto = pred == y
        limiares = np.round(np.arange(0.30, 0.96, 0.05), 2)
        curva = [{"limiar": float(t), "cobertura": float((conf >= t).mean()),
                  "acuracia_aceitos": float(acerto[conf >= t].mean()) if (conf >= t).any() else None}
                 for t in limiares]
        alta = conf >= config.LIMIAR_CONFIANCA_ALTA
        mod = (conf >= config.LIMIAR_CONFIANCA_MINIMA) & ~alta
        baixa = conf < config.LIMIAR_CONFIANCA_MINIMA
        faixas = {nome: {"fracao": float(mk.mean()),
                         "acuracia": float(acerto[mk].mean()) if mk.any() else None}
                  for nome, mk in (("alta", alta), ("moderada", mod), ("baixa", baixa))}
        bins = np.linspace(0, 1, 11)
        ece, conf_b, acc_b = 0.0, [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.any():
                ece += mk.mean() * abs(acerto[mk].mean() - conf[mk].mean())
                conf_b.append(conf[mk].mean())
                acc_b.append(acerto[mk].mean())
        out[particao] = {"acuracia": float(acerto.mean()), "ece": float(ece),
                         "faixas": faixas, "curva": curva,
                         "confianca_media_acertos": float(conf[acerto].mean()),
                         "confianca_media_erros": float(conf[~acerto].mean())}
        rot = "Validação" if particao == "validacao" else "Teste"
        fina = np.round(np.arange(0.30, 0.99, 0.01), 2)
        fina = [t for t in fina if (conf >= t).sum() >= 10]
        axs[0].plot([(conf >= t).mean() for t in fina], [acerto[conf >= t].mean() for t in fina],
                    "-", lw=1.4, color=cor, label=rot)
        axs[1].plot(conf_b, acc_b, "o-", ms=3, color=cor, label=rot)
        if particao == "teste":
            for t in (config.LIMIAR_CONFIANCA_MINIMA, config.LIMIAR_CONFIANCA_ALTA):
                cx, cy = (conf >= t).mean(), acerto[conf >= t].mean()
                axs[0].plot(cx, cy, "o", ms=5, color="#e07a5f", zorder=5)
                axs[0].annotate(f"τ={t:.2f}", (cx, cy), textcoords="offset points",
                                xytext=(-36, -12), fontsize=7)
        print(f"E1 {particao}: acc={acerto.mean():.3f} ECE={ece:.3f} faixas={faixas}")
    axs[0].set_xlabel("Cobertura (fração aceita)")
    axs[0].set_ylabel("Acurácia dos aceitos")
    axs[0].legend(fontsize=7, frameon=False)
    axs[0].set_title("(a) Classificação seletiva", fontsize=9)
    axs[1].plot([0, 1], [0, 1], "--", color="#aaa", lw=0.8)
    axs[1].set_xlabel("Confiança média")
    axs[1].set_ylabel("Acurácia")
    axs[1].set_title("(b) Diagrama de confiabilidade", fontsize=9)
    fig.tight_layout()
    fig.savefig(G / "visao_confianca.png")
    plt.close(fig)
    return out


def e2_robustez(d, pac, n_por_classe=None):
    teste = pd.read_csv(config.ARQ_DIVISAO).query("particao == 'teste'")
    imgs = [cv2.imread(str(config.RAIZ / c)) for c in teste["caminho"]]
    y = teste["classe"].values
    ext = ExtratorMobileNet()
    vis = json.load(open(config.DIR_METRICAS / "visao.json", encoding="utf-8"))
    from src.machine_learning.modelos import candidatos
    nome_cls = max([n for n, v in vis["modelos"].items() if v["caracteristica"] == "classica"],
                   key=lambda n: vis["modelos"][n]["val_f1_macro"])
    _, _, pipe_cls, _ = candidatos()[nome_cls]
    pipe_cls.set_params(**vis["modelos"][nome_cls]["melhores_hiperparametros"])
    mtr = d["particao"] == "treino"
    pipe_cls.fit(d["classica"][mtr], d["classe"][mtr])
    res = {}
    for nome, f in pp.PERTURBACOES.items():
        alteradas = [f(im) for im in imgs]
        acc_cnn = float((pac["modelo"].predict(ext.extrair(alteradas)) == y).mean())
        acc_cls = float((pipe_cls.predict(extrair_lote(alteradas)) == y).mean())
        res[nome] = {"cnn": acc_cnn, "classico": acc_cls}
        print(f"E2 {nome:16s} CNN={acc_cnn:.3f} clássico={acc_cls:.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(len(res))
    ax.bar(x - 0.2, [v["cnn"] for v in res.values()], 0.4, color=AZUL, label=pac["nome"])
    ax.bar(x + 0.2, [v["classico"] for v in res.values()], 0.4, color="#e07a5f", label=nome_cls)
    for xi, v in zip(x, res.values()):
        ax.text(xi - 0.2, v["cnn"] + 0.01, f"{v['cnn']:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in res], fontsize=7)
    ax.set_ylabel("Acurácia (teste)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.savefig(G / "visao_robustez.png")
    plt.close(fig)
    base = imgs[int(np.where(y == "plastic")[0][3])]
    fig, axs = plt.subplots(1, len(pp.PERTURBACOES), figsize=(7.2, 1.3))
    for ax, (nome, f) in zip(axs, pp.PERTURBACOES.items()):
        ax.imshow(cv2.cvtColor(f(base), cv2.COLOR_BGR2RGB))
        ax.set_title(nome, fontsize=6)
        ax.axis("off")
    fig.savefig(G / "visao_perturbacoes_exemplo.png")
    plt.close(fig)
    return {"modelo_classico_comparado": nome_cls, "acuracias": res}


ROTULOS = {"plastic": ["PET", "PEAD", "PP", "PLASTICO", "GARRAFA PET"],
           "metal": ["ALUMINIO", "LATA DE ACO", "LATA"],
           "glass": ["VIDRO", "VIDRO RETORNAVEL"],
           "paper": ["PAPEL", "PAPEL RECICLAVEL"],
           "cardboard": ["PAPELAO", "CAIXA DE PAPELAO"]}
FONTES = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]


def aplicar_rotulo(img, texto, rng):
    """Desenha uma etiqueta retangular com texto e a cola, rotacionada, na imagem."""
    fonte = FONTES[rng.integers(len(FONTES))]
    escala = float(rng.uniform(0.9, 1.6))
    esp = 2 if escala < 1.3 else 3
    (tw, th), base = cv2.getTextSize(texto, fonte, escala, esp)
    pad = 12
    etiqueta = np.full((th + base + 2 * pad, tw + 2 * pad, 3), 245, np.uint8)
    cv2.putText(etiqueta, texto, (pad, pad + th), fonte, escala, (20, 20, 20), esp, cv2.LINE_AA)
    ang = float(rng.uniform(-8, 8))
    h, w = etiqueta.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    rot = cv2.warpAffine(etiqueta, M, (w, h), borderValue=(0, 0, 0))
    mascara = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h)) > 0
    H, W = img.shape[:2]
    y0 = int(rng.integers(10, max(11, H - h - 10)))
    x0 = int(rng.integers(10, max(11, W - w - 10)))
    out = img.copy()
    regiao = out[y0:y0 + h, x0:x0 + w]
    regiao[mascara] = rot[mascara]
    return out, (x0, y0, w, h)


def cer(ref, hip):
    ref, hip = normalizar(ref).replace(" ", ""), normalizar(hip).replace(" ", "")
    dp = list(range(len(hip) + 1))
    for i, a in enumerate(ref, 1):
        ant, dp[0] = dp[0], i
        for j, b in enumerate(hip, 1):
            ant, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, ant + (a != b))
    return dp[-1] / max(1, len(ref))


def decidir_material(visual, ocr_mats, servicos):
    fatos = {"texto": "", "intencao": "identificar_material", "intencao_incerta": False,
             "ranking_intencao": [], "material_texto": None, "bairro": None,
             "visual": visual, "ocr_materiais": ocr_mats, **servicos}
    f, rastro = MotorRegras(REGRAS).executar(fatos)
    return f.get("material"), [r.split(":")[0] for r in rastro]


def e3_ocr(pac, servicos):
    rng = np.random.default_rng(config.SEMENTE)
    teste = pd.read_csv(config.ARQ_DIVISAO).query("particao == 'teste' and classe != 'trash'")
    ext = ExtratorMobileNet()
    linhas = []
    exemplos = []
    for _, r in teste.iterrows():
        img = cv2.imread(str(config.RAIZ / r["caminho"]))
        texto = ROTULOS[r["classe"]][rng.integers(len(ROTULOS[r["classe"]]))]
        com, (x, y, w, h) = aplicar_rotulo(img, texto, rng)
        ocr_recorte = ocr.ler_texto(com[y:y + h, x:x + w], psm=7)["texto"]
        ocr_cena = ocr.ler_texto(com)["texto"]
        ocr_original = ocr.ler_texto(img)["texto"]
        P = pac["modelo"].predict_proba(ext.extrair(com))[0]
        ordem = np.argsort(P)[::-1]
        visual = {"material": MAPA_MATERIAL[pac["modelo"].classes_[ordem[0]]],
                  "confianca": float(P[ordem[0]]),
                  "top": [(MAPA_MATERIAL[pac["modelo"].classes_[i]], float(P[i])) for i in ordem]}
        verdade = MAPA_MATERIAL[r["classe"]]
        mats_cena = materiais_em_rotulo(ocr_cena)
        m_vis, _ = decidir_material(visual, [], servicos)
        m_fus, rastro = decidir_material(visual, mats_cena, servicos)
        linhas.append({"caminho": r["caminho"], "classe": r["classe"], "rotulo": texto,
                       "ocr_recorte": ocr_recorte, "ocr_cena": ocr_cena,
                       "cer_recorte": cer(texto, ocr_recorte),
                       "material_ocr_correto": bool(mats_cena) and mats_cena[0] == verdade,
                       "ocr_achou_material": bool(mats_cena),
                       "falso_material_sem_rotulo": bool(materiais_em_rotulo(ocr_original)),
                       "conf_visual": visual["confianca"],
                       "visual_correto": visual["material"] == verdade,
                       "decisao_so_imagem": m_vis, "decisao_fusao": m_fus, "verdade": verdade,
                       "usou_R03": "R03" in rastro})
        if (len(exemplos) < 4 and "R03" in rastro and m_fus == verdade
                and visual["material"] != verdade
                and r["classe"] not in {e[4] for e in exemplos}):
            exemplos.append((com, texto, visual, m_fus, r["classe"]))
    df = pd.DataFrame(linhas)
    df.to_csv(config.DIR_METRICAS / "ocr_rotulos_sinteticos.csv", index=False)

    def taxa(col_dec):
        acerto = (df[col_dec] == df["verdade"]).mean()
        abst = df[col_dec].isna().mean()
        return {"acerto": float(acerto), "abstencao": float(abst),
                "erro": float(1 - acerto - abst)}
    res = {"n": len(df), "cer_medio_recorte": float(df["cer_recorte"].mean()),
           "acerto_exato_recorte": float((df["cer_recorte"] == 0).mean()),
           "ocr_cena_detectou_material": float(df["ocr_achou_material"].mean()),
           "ocr_cena_material_correto": float(df["material_ocr_correto"].mean()),
           "precisao_material_ocr": float(df["material_ocr_correto"].sum()
                                          / max(1, df["ocr_achou_material"].sum())),
           "falso_material_em_imagens_sem_rotulo": float(df["falso_material_sem_rotulo"].mean()),
           "acuracia_visual_com_rotulo": float(df["visual_correto"].mean()),
           "decisao_so_imagem": taxa("decisao_so_imagem"),
           "decisao_fusao": taxa("decisao_fusao"),
           "casos_regra_R03": int(df["usou_R03"].sum())}
    print("E3", json.dumps(res, ensure_ascii=False))
    if exemplos:
        fig, axs = plt.subplots(1, len(exemplos), figsize=(7.2, 1.95))
        for ax, (im, texto, vis, mf, _) in zip(np.atleast_1d(axs), exemplos):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(f"visão: {vis['material']} ({vis['confianca']:.0%})\n"
                         f"rótulo OCR → {mf}", fontsize=6.5)
            ax.axis("off")
        fig.savefig(G / "ocr_fusao_exemplos.png")
        plt.close(fig)
    return res


def e4_busca():
    mapa = MapaColeta()
    materiais = ["papel", "papelao", "plastico", "metal", "vidro", "perigoso", "oleo", "organico"]
    algs = {"BFS": mapa.bfs, "Custo uniforme": mapa.custo_uniforme,
            "Gulosa": mapa.gulosa, "A*": mapa.a_estrela}
    linhas = []
    for inicio, mat in itertools.product(mapa.coord, materiais):
        otimo = mapa.custo_uniforme(inicio, mat).custo_km
        for nome, alg in algs.items():
            t = time.perf_counter()
            for _ in range(20):
                r = alg(inicio, mat)
            dt = (time.perf_counter() - t) / 20 * 1e6
            linhas.append({"algoritmo": nome, "inicio": inicio, "material": mat,
                           "expandidos": r.nos_expandidos, "custo": r.custo_km,
                           "otimo": abs(r.custo_km - otimo) < 1e-9,
                           "razao_custo": r.custo_km / otimo if otimo > 0 else 1.0,
                           "tempo_us": dt})
    df = pd.DataFrame(linhas)
    resumo = df.groupby("algoritmo").agg(expandidos_medio=("expandidos", "mean"),
                                         pct_otimo=("otimo", "mean"),
                                         razao_custo_media=("razao_custo", "mean"),
                                         razao_custo_max=("razao_custo", "max"),
                                         tempo_medio_us=("tempo_us", "mean")).reindex(list(algs))
    print("E4\n", resumo.round(3))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))
    axs[0].bar(resumo.index, resumo["expandidos_medio"], color=CORES[:4])
    axs[0].set_ylabel("Nós expandidos (média)")
    axs[1].bar(resumo.index, resumo["razao_custo_media"], color=CORES[:4])
    axs[1].set_ylim(0.95, max(1.1, resumo["razao_custo_media"].max() + 0.03))
    axs[1].set_ylabel("Custo / custo ótimo")
    for ax, col, fmt in ((axs[0], "expandidos_medio", "{:.2f}"), (axs[1], "razao_custo_media", "{:.3f}")):
        for i, v in enumerate(resumo[col]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5)
    for ax in axs:
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(G / "busca_comparacao.png")
    plt.close(fig)
    r = mapa.a_estrela("Ipioca", "perigoso")
    fig, ax = plt.subplots(figsize=(3.3, 3.4))
    for a, viz in mapa.adj.items():
        for b in viz:
            ax.plot([mapa.coord[a][1], mapa.coord[b][1]], [mapa.coord[a][0], mapa.coord[b][0]],
                    color="#cfcfcf", lw=0.7, zorder=1)
    for nome, (la, lo) in mapa.coord.items():
        ax.scatter(lo, la, s=8, color="#777", zorder=2)
    bairros_pontos = {p["bairro"] for p in mapa.pontos}
    for b in bairros_pontos:
        ax.scatter(mapa.coord[b][1], mapa.coord[b][0], s=40, marker="s", color="#81b29a", zorder=3)
    xs = [mapa.coord[b][1] for b in r.caminho]
    ys = [mapa.coord[b][0] for b in r.caminho]
    ax.plot(xs, ys, color="#e07a5f", lw=2.2, zorder=4)
    ax.annotate("Ipioca", (xs[0], ys[0]), fontsize=7, xytext=(-30, 4), textcoords="offset points")
    ax.annotate("Centro", (xs[-1], ys[-1]), fontsize=7, xytext=(-28, -10), textcoords="offset points")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(G / "busca_grafo_rota.png")
    plt.close(fig)
    return {"n_consultas": int(len(df) / len(algs)), "n_bairros": len(mapa.coord),
            "n_arestas": int(sum(len(v) for v in mapa.adj.values()) / 2),
            "resumo": resumo.round(4).to_dict("index"),
            "exemplo_ipioca_perigoso": {"caminho": r.caminho, "custo_km": r.custo_km,
                                        "expandidos": r.nos_expandidos}}


def e5_nao_supervisionado(d):
    m = d["particao"] == "teste"
    X, y = d["cnn"][m], d["classe"][m]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    res = {}
    for nome, Z in (("cnn", Xn), ("classica", (d["classica"][m] - d["classica"][m].mean(0))
                                  / (d["classica"][m].std(0) + 1e-9))):
        km = KMeans(6, n_init=10, random_state=config.SEMENTE).fit(Z)
        res[nome] = {"ARI": float(adjusted_rand_score(y, km.labels_)),
                     "NMI": float(normalized_mutual_info_score(y, km.labels_))}
    print("E5", res)
    emb = TSNE(2, random_state=config.SEMENTE, perplexity=30, init="pca").fit_transform(Xn)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for c, cor, nome in zip(config.CLASSES, CORES, NOMES):
        k = y == c
        ax.scatter(emb[k, 0], emb[k, 1], s=6, color=cor, label=nome, alpha=0.8)
    ax.legend(fontsize=6, markerscale=1.5, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    fig.savefig(G / "visao_tsne_embeddings.png")
    plt.close(fig)
    return res


def e6_ponta_a_ponta():
    from src.integracao.pipeline import EcoTriagem
    rnd = random.Random(config.SEMENTE)
    eco = EcoTriagem()
    k = kb()
    perguntas = pd.read_csv(config.DIR_PLN / "teste_manual.csv").fillna("")
    alvo = ["identificar_material", "como_descartar", "reciclavel", "preparo_descarte",
            "tempo_decomposicao"]
    perguntas = perguntas[perguntas["intencao"].isin(alvo) & (perguntas["material"] == "")]
    teste = pd.read_csv(config.ARQ_DIVISAO).query("particao == 'teste'")
    linhas, tempos_img, tempos_txt = [], [], []
    for _, r in teste.iterrows():
        eco.reiniciar()
        a = eco.analisar_imagem(config.RAIZ / r["caminho"])
        tempos_img.append(a["tempo_total_ms"])
        q = perguntas.iloc[rnd.randrange(len(perguntas))]
        t = time.time()
        resp = eco.perguntar(q["texto"])
        tempos_txt.append((time.time() - t) * 1000)
        verdade = MAPA_MATERIAL[r["classe"]]
        mat = resp["material"]
        linhas.append({
            "imagem": r["caminho"], "classe": r["classe"], "pergunta": q["texto"],
            "intencao_esperada": q["intencao"], "intencao_prevista": resp["intencao"],
            "material_verdadeiro": verdade, "material_decidido": mat,
            "confianca_visual": a["confianca"],
            "intencao_ok": resp["intencao"] == q["intencao"],
            "material_ok": mat == verdade, "abstencao": mat is None,
            "lixeira_ok": (mat is not None and k[mat]["lixeira"] == k[verdade]["lixeira"]),
        })
    df = pd.DataFrame(linhas)
    df.to_csv(config.DIR_METRICAS / "ponta_a_ponta.csv", index=False)
    resp_ok = df["intencao_ok"] & df["material_ok"]
    res = {"n_cenarios": len(df), "n_perguntas_distintas": int(df["pergunta"].nunique()),
           "intencao_correta": float(df["intencao_ok"].mean()),
           "material_correto": float(df["material_ok"].mean()),
           "abstencao_pediu_nova_foto": float(df["abstencao"].mean()),
           "material_errado_sem_abstencao": float((~df["material_ok"] & ~df["abstencao"]).mean()),
           "lixeira_correta": float(df["lixeira_ok"].mean()),
           "acuracia_quando_respondeu": float(df.loc[~df["abstencao"], "material_ok"].mean()),
           "resposta_totalmente_correta": float(resp_ok.mean()),
           "latencia_imagem_ms_mediana": float(np.median(tempos_img)),
           "latencia_imagem_ms_p95": float(np.percentile(tempos_img, 95)),
           "latencia_pergunta_ms_mediana": float(np.median(tempos_txt))}
    print("E6", json.dumps(res, ensure_ascii=False))
    return res


def e7_exemplos(d, pac):
    m = d["particao"] == "teste"
    P = pac["modelo"].predict_proba(d[pac["caracteristica"]][m])
    classes = np.array(pac["modelo"].classes_)
    pred, conf = classes[P.argmax(1)], P.max(1)
    y, cam = d["classe"][m], d["caminho"][m]
    rng = np.random.default_rng(config.SEMENTE)
    certos = [rng.choice(np.where((y == c) & (pred == y))[0]) for c in config.CLASSES]
    erros = rng.choice(np.where(pred != y)[0], 6, replace=False)
    for nome, idx, titulo in (("exemplos_entrada_saida.png", certos, "acerto"),
                              ("visao_erros.png", erros, "erro")):
        fig, axs = plt.subplots(1, 6, figsize=(7.2, 1.65))
        for ax, i in zip(axs, idx):
            ax.imshow(cv2.cvtColor(cv2.imread(str(config.RAIZ / cam[i])), cv2.COLOR_BGR2RGB))
            ax.set_title(f"real: {config.CLASSES_PT[y[i]]}\nprev.: {config.CLASSES_PT[pred[i]]}"
                         f" ({conf[i]:.0%})", fontsize=6.5,
                         color="#1b4332" if titulo == "acerto" else "#9d0208")
            ax.axis("off")
        destino = (config.DIR_EXEMPLOS if titulo == "acerto" else G) / nome
        fig.savefig(destino)
        plt.close(fig)
    return {"exemplos_erros": [{"imagem": str(cam[i]), "real": str(y[i]), "previsto": str(pred[i]),
                                "confianca": float(conf[i])} for i in erros]}


def main():
    pedidos = set(sys.argv[1:]) or {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}
    res = json.load(open(ARQ_SAIDA, encoding="utf-8")) if ARQ_SAIDA.exists() else {}
    d, pac = carregar_cache(), pacote_visual()
    servicos = {"_kb": kb(), "_mapa": MapaColeta(), "_faq": None}
    passos = {"E1": lambda: e1_confianca(d, pac), "E2": lambda: e2_robustez(d, pac),
              "E3": lambda: e3_ocr(pac, servicos), "E4": e4_busca,
              "E5": lambda: e5_nao_supervisionado(d), "E6": e6_ponta_a_ponta,
              "E7": lambda: e7_exemplos(d, pac)}
    for chave in sorted(pedidos):
        t = time.time()
        res[chave] = passos[chave]()
        print(f"[{chave}] {time.time() - t:.0f}s")
        salvar_json(res, ARQ_SAIDA)


if __name__ == "__main__":
    main()
