"""Extrai e armazena em cache as características de todas as imagens."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.visao_computacional.extrator_classico import extrair_lote
from src.visao_computacional.extrator_cnn import ExtratorMobileNet


def main():
    df = pd.read_csv(config.ARQ_DIVISAO)
    imagens = [cv2.imread(str(config.RAIZ / c)) for c in df["caminho"]]
    ext = ExtratorMobileNet()
    t = time.time()
    cnn = ext.extrair(imagens)
    cnn_esp = ext.extrair([cv2.flip(im, 1) for im in imagens])
    t_cnn = (time.time() - t) / (2 * len(imagens))
    t = time.time()
    classica = extrair_lote(imagens)
    t_cls = (time.time() - t) / len(imagens)
    config.DIR_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.DIR_CACHE / "caracteristicas.npz", cnn=cnn,
                        cnn_espelhada=cnn_esp, classica=classica,
                        caminho=df["caminho"].values, classe=df["classe"].values,
                        particao=df["particao"].values, grupo=df["grupo"].values,
                        tempo_cnn_ms=t_cnn * 1000, tempo_classica_ms=t_cls * 1000)
    print(f"{len(df)} imagens | CNN {t_cnn * 1000:.1f} ms/img | "
          f"clássica {t_cls * 1000:.1f} ms/img")


if __name__ == "__main__":
    main()
