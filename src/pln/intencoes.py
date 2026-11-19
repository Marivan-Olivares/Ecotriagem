"""Reconhecimento de intenções (classificação supervisionada de textos)."""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.pln.preprocessamento import normalizar

INTENCOES = ["identificar_material", "como_descartar", "ponto_coleta",
             "preparo_descarte", "reciclavel", "tempo_decomposicao",
             "duvida_geral", "saudacao_ajuda", "fora_escopo"]

LIMIAR_INTENCAO = 0.35


def vetorizador(usar_caracteres: bool = True):
    palavras = TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2),
                               sublinear_tf=True, min_df=1)
    if not usar_caracteres:
        return palavras
    caracteres = TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                 ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    return FeatureUnion([("palavras", palavras), ("caracteres", caracteres)])


def criar_modelo(nome: str = "svm_linear", usar_caracteres: bool = True) -> Pipeline:
    if nome == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif nome == "regressao_logistica":
        clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
    elif nome == "svm_linear":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"),
                                     method="sigmoid", cv=5)
    else:
        raise ValueError(nome)
    return Pipeline([("tfidf", vetorizador(usar_caracteres)), ("clf", clf)])


@dataclass
class ResultadoIntencao:
    intencao: str
    confianca: float
    ranking: list

    @property
    def incerta(self) -> bool:
        return self.confianca < LIMIAR_INTENCAO


class ClassificadorIntencao:
    def __init__(self, caminho=config.ARQ_MODELO_INTENCAO):
        self.modelo = joblib.load(caminho)

    def prever(self, texto: str) -> ResultadoIntencao:
        p = self.modelo.predict_proba([texto])[0]
        classes = self.modelo.classes_
        ordem = np.argsort(p)[::-1]
        return ResultadoIntencao(intencao=str(classes[ordem[0]]),
                                 confianca=float(p[ordem[0]]),
                                 ranking=[(str(classes[i]), float(p[i])) for i in ordem[:3]])
