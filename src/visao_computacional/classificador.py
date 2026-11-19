"""Classificador visual de resíduos usado em produção (pipeline integrado)."""
from dataclasses import dataclass

import joblib
import numpy as np

from src import config
from src.visao_computacional.extrator_classico import caracteristicas_classicas
from src.visao_computacional.extrator_cnn import ExtratorMobileNet

MAPA_MATERIAL = {"cardboard": "papelao", "glass": "vidro", "metal": "metal",
                 "paper": "papel", "plastic": "plastico", "trash": "rejeito"}


@dataclass
class ResultadoVisual:
    classe: str
    material: str
    confianca: float
    probabilidades: dict

    def top(self, k=3):
        return sorted(self.probabilidades.items(), key=lambda x: -x[1])[:k]


class ClassificadorVisual:
    def __init__(self, caminho=config.ARQ_MODELO_VISAO):
        pacote = joblib.load(caminho)
        self.modelo = pacote["modelo"]
        self.caracteristica = pacote["caracteristica"]
        self.nome = pacote["nome"]
        self.extrator = ExtratorMobileNet() if self.caracteristica == "cnn" else None

    def caracteristicas(self, img_bgr) -> np.ndarray:
        if self.caracteristica == "cnn":
            return self.extrator.extrair(img_bgr)
        return caracteristicas_classicas(img_bgr)[None, :]

    def classificar(self, img_bgr) -> ResultadoVisual:
        p = self.modelo.predict_proba(self.caracteristicas(img_bgr))[0]
        classes = list(self.modelo.classes_)
        i = int(np.argmax(p))
        return ResultadoVisual(classe=classes[i], material=MAPA_MATERIAL[classes[i]],
                               confianca=float(p[i]),
                               probabilidades={c: float(v) for c, v in zip(classes, p)})
