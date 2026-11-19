"""Preparação, limpeza e divisão do dataset TrashNet."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

LIMIAR_HAMMING = 40
LIMIAR_MAD = 5.0


def dhash(img_bgr: np.ndarray, tamanho: int = 16) -> np.ndarray:
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    red = cv2.resize(cinza, (tamanho + 1, tamanho), interpolation=cv2.INTER_AREA)
    return (red[:, 1:] > red[:, :-1]).flatten().astype(np.uint8)


def miniatura(img_bgr: np.ndarray) -> np.ndarray:
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cinza, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


class UniaoBusca:
    def __init__(self, n):
        self.pai = list(range(n))

    def achar(self, x):
        while self.pai[x] != x:
            self.pai[x] = self.pai[self.pai[x]]
            x = self.pai[x]
        return x

    def unir(self, a, b):
        self.pai[self.achar(a)] = self.achar(b)


def main() -> None:
    linhas, descartadas, hashes, minis = [], [], [], []
    for classe in config.CLASSES:
        for arq in sorted((config.DIR_IMAGENS / classe).glob("*.jpg")):
            img = cv2.imread(str(arq))
            if img is None or img.ndim != 3 or min(img.shape[:2]) < 64:
                descartadas.append(str(arq.name))
                continue
            linhas.append({
                "caminho": str(arq.relative_to(config.RAIZ)),
                "classe": classe,
                "altura": img.shape[0], "largura": img.shape[1],
                "brilho_medio": float(img.mean()),
            })
            hashes.append(dhash(img))
            minis.append(miniatura(img))
    df = pd.DataFrame(linhas)
    total_validas = len(df)
    print(f"imagens válidas: {len(df)} | descartadas: {len(descartadas)}")

    bits = np.stack(hashes)
    uf = UniaoBusca(len(df))
    pares_dup, conflitos = 0, []
    for i in range(len(df)):
        dist = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(dist <= LIMIAR_HAMMING)[0] + i + 1:
            j = int(j)
            if np.abs(minis[i] - minis[j]).mean() >= LIMIAR_MAD:
                continue
            pares_dup += 1
            if df.at[i, "classe"] != df.at[j, "classe"]:
                conflitos.append((df.at[i, "caminho"], df.at[j, "caminho"]))
            uf.unir(i, j)
    pares_conflito = len(conflitos)
    df["grupo"] = [uf.achar(i) for i in range(len(df))]
    grupos_conflito = df.groupby("grupo")["classe"].nunique()
    grupos_conflito = set(grupos_conflito[grupos_conflito > 1].index)
    removidas = df[df["grupo"].isin(grupos_conflito)]["caminho"].tolist()
    df = df[~df["grupo"].isin(grupos_conflito)].reset_index(drop=True)
    n_grupos = df["grupo"].nunique()
    print(f"removidas por rótulo conflitante: {len(removidas)}")
    print(f"pares quase-duplicados: {pares_dup} (classes conflitantes: "
          f"{pares_conflito}); grupos: {n_grupos}")

    grupos = (df.groupby("grupo")["classe"]
                .agg(lambda s: s.value_counts().index[0]).reset_index())
    g_treino, g_resto = train_test_split(
        grupos, test_size=0.30, stratify=grupos["classe"],
        random_state=config.SEMENTE)
    g_val, g_teste = train_test_split(
        g_resto, test_size=0.50, stratify=g_resto["classe"],
        random_state=config.SEMENTE)
    mapa = {**{g: "treino" for g in g_treino["grupo"]},
            **{g: "validacao" for g in g_val["grupo"]},
            **{g: "teste" for g in g_teste["grupo"]}}
    df["particao"] = df["grupo"].map(mapa)

    config.DIR_PROCESSADO.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ARQ_DIVISAO, index=False)

    tabela = pd.crosstab(df["classe"], df["particao"])[["treino", "validacao", "teste"]]
    tabela.loc["total"] = tabela.sum()
    print(tabela)

    relatorio = {
        "total_imagens_validas": int(total_validas),
        "imagens_descartadas": descartadas,
        "resolucoes": df.groupby(["largura", "altura"]).size()
                        .rename("n").reset_index().to_dict("records"),
        "limiar_hamming_dhash256": LIMIAR_HAMMING,
        "limiar_mad": LIMIAR_MAD,
        "pares_conflitantes": conflitos,
        "imagens_removidas_rotulo_conflitante": removidas,
        "total_imagens_apos_limpeza": int(len(df)),
        "pares_quase_duplicados": pares_dup,
        "pares_duplicados_com_classes_diferentes": pares_conflito,
        "imagens_em_grupos_com_duplicata": int(
            (df.groupby("grupo")["grupo"].transform("size") > 1).sum()),
        "grupos": int(n_grupos),
        "distribuicao": {c: tabela.loc[c].to_dict() for c in tabela.index},
    }
    config.DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    with open(config.DIR_METRICAS / "qualidade_dados.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
