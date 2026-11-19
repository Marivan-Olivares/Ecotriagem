"""Fluxo integrado EcoTriagem: Visão Computacional + PLN + ML + Decisão."""
import json
import time

from src import config
from src.decisao.agente import AgenteEcoTriagem
from src.decisao.busca import MapaColeta
from src.pln.entidades import ExtratorEntidades, materiais_em_rotulo
from src.pln.faq import BuscaFAQ
from src.pln.intencoes import ClassificadorIntencao
from src.visao_computacional import ocr
from src.visao_computacional.classificador import ClassificadorVisual
from src.visao_computacional.preprocessamento import carregar_imagem, padronizar


class EcoTriagem:
    def __init__(self, usar_ocr: bool = True):
        self.visao = ClassificadorVisual()
        self.usar_ocr = usar_ocr and ocr.disponivel()
        kb = {k: v for k, v in json.load(open(config.DIR_CONHECIMENTO / "materiais.json",
                                              encoding="utf-8")).items()
              if not k.startswith("_")}
        self.kb = kb
        self.mapa = MapaColeta()
        self.agente = AgenteEcoTriagem(ClassificadorIntencao(), ExtratorEntidades(),
                                       kb, self.mapa, BuscaFAQ())

    def reiniciar(self):
        self.agente.reiniciar()

    def analisar_imagem(self, origem) -> dict:
        t0 = time.time()
        img = padronizar(carregar_imagem(origem))
        r = self.visao.classificar(img)
        t_visao = time.time() - t0
        texto_ocr, mats_ocr = "", []
        if self.usar_ocr:
            texto_ocr = ocr.ler_texto(img)["texto"]
            mats_ocr = materiais_em_rotulo(texto_ocr)
        from src.visao_computacional.classificador import MAPA_MATERIAL
        top = [(MAPA_MATERIAL[c], p) for c, p in r.top(6)]
        visual = {"classe": r.classe, "material": r.material, "confianca": r.confianca,
                  "top": top}
        self.agente.perceber_imagem(visual, texto_ocr, mats_ocr)
        return {**visual, "material_pt": self.kb[r.material]["nome"],
                "ocr_texto": texto_ocr, "ocr_materiais": mats_ocr,
                "tempo_visao_ms": round(t_visao * 1000, 1),
                "tempo_total_ms": round((time.time() - t0) * 1000, 1)}

    def perguntar(self, texto: str) -> dict:
        return self.agente.responder(texto)

    def processar(self, imagem=None, pergunta: str = "O que é isso e onde descarto?") -> dict:
        analise = self.analisar_imagem(imagem) if imagem is not None else None
        return {"imagem": analise, "resposta": self.perguntar(pergunta)}
