"""Perguntas e respostas por recuperação (busca semântica leve)."""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.pln.preprocessamento import normalizar

LIMIAR_FAQ = 0.30


class BuscaFAQ:
    def __init__(self):
        self.base = json.load(open(config.DIR_CONHECIMENTO / "faq.json", encoding="utf-8"))
        self.vet = TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                   ngram_range=(3, 5), sublinear_tf=True)
        self.matriz = self.vet.fit_transform([b["pergunta"] for b in self.base])

    def buscar(self, pergunta: str, k: int = 1):
        sims = cosine_similarity(self.vet.transform([pergunta]), self.matriz)[0]
        ordem = sims.argsort()[::-1][:k]
        return [(self.base[i]["pergunta"], self.base[i]["resposta"], float(sims[i]))
                for i in ordem]

    def responder(self, pergunta: str):
        perg, resp, sim = self.buscar(pergunta, 1)[0]
        if sim < LIMIAR_FAQ:
            return None, sim, perg
        return resp, sim, perg
