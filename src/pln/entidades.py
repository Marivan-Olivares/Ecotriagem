"""Extração de entidades: materiais/objetos, bairros e códigos de reciclagem."""
import difflib
import json
import re
from dataclasses import dataclass, field

from src import config
from src.pln.preprocessamento import singularizar, tokens_singulares, normalizar

LIMIAR_FUZZY = 0.85
TAMANHO_MIN_FUZZY = 6
NAO_APROXIMAR = {"material", "materiai", "embalagem", "objeto", "imagem", "residuo",
                 "reciclavel", "reciclagem", "reciclado", "previsao", "descartar",
                 "descarte", "lixeira", "coleta", "decompor", "fotografia"}
MAX_NGRAMA = 4

CODIGOS_RECICLAGEM = {
    "pet": "plastico", "pead": "plastico", "hdpe": "plastico", "pvc": "plastico",
    "pebd": "plastico", "ldpe": "plastico", "pp": "plastico", "ps": "plastico",
    "aluminio": "metal", "aluminum": "metal", "aco": "metal", "steel": "metal",
    "lata": "metal", "vidro": "vidro", "glass": "vidro", "papel": "papel",
    "paper": "papel", "papelao": "papelao", "cardboard": "papelao",
    "plastico": "plastico", "plastic": "plastico", "reciclavel": None,
}


@dataclass
class Entidades:
    materiais: list = field(default_factory=list)
    bairro: str | None = None

    @property
    def material(self):
        return self.materiais[0][0] if self.materiais else None


class ExtratorEntidades:
    def __init__(self):
        lex = json.load(open(config.DIR_CONHECIMENTO / "objetos.json", encoding="utf-8"))
        self.lexico = {}
        for material, termos in lex.items():
            if material.startswith("_"):
                continue
            for termo in termos:
                chave = " ".join(tokens_singulares(termo))
                self.lexico.setdefault(chave, material)
        self.palavras_simples = [k for k in self.lexico
                                 if " " not in k and len(k) >= TAMANHO_MIN_FUZZY - 1]

        mapa = json.load(open(config.DIR_CONHECIMENTO / "mapa_maceio.json", encoding="utf-8"))
        self.bairros = {normalizar(b): b for b in mapa["bairros"]}
        apelidos = {"tabuleiro": "Tabuleiro do Martins", "gruta": "Gruta de Lourdes",
                    "benedito": "Benedito Bentes", "trapiche": "Trapiche da Barra",
                    "ufal": "Cidade Universitária", "cidade universitaria": "Cidade Universitária",
                    "jatiuca": "Jatiúca", "pajucara": "Pajuçara", "garca torta": "Garça Torta"}
        self.bairros.update(apelidos)

    def extrair_materiais(self, texto: str) -> list:
        toks = tokens_singulares(texto)
        achados, ocupado = [], [False] * len(toks)
        for n in range(MAX_NGRAMA, 0, -1):
            for i in range(len(toks) - n + 1):
                if any(ocupado[i:i + n]):
                    continue
                ngrama = " ".join(toks[i:i + n])
                if ngrama in self.lexico:
                    achados.append((self.lexico[ngrama], ngrama, i))
                    ocupado[i:i + n] = [True] * n
        for i, tok in enumerate(toks):
            if ocupado[i] or len(tok) < TAMANHO_MIN_FUZZY or tok in NAO_APROXIMAR:
                continue
            candidatos = [p for p in self.palavras_simples if p[0] == tok[0]]
            parecido = difflib.get_close_matches(tok, candidatos, n=1,
                                                 cutoff=LIMIAR_FUZZY)
            if parecido:
                achados.append((self.lexico[parecido[0]], parecido[0], i))
                ocupado[i] = True
        return sorted(achados, key=lambda x: x[2])

    def extrair_bairro(self, texto: str) -> str | None:
        t = f" {normalizar(texto)} "
        melhores = sorted(self.bairros, key=len, reverse=True)
        for chave in melhores:
            if f" {chave} " in t:
                return self.bairros[chave]
        for tok in t.split():
            if len(tok) >= 6:
                p = difflib.get_close_matches(tok, [b for b in self.bairros if " " not in b],
                                              n=1, cutoff=0.85)
                if p:
                    return self.bairros[p[0]]
        return None

    def extrair(self, texto: str) -> Entidades:
        return Entidades(materiais=self.extrair_materiais(texto),
                         bairro=self.extrair_bairro(texto))


def materiais_em_rotulo(texto_ocr: str) -> list[str]:
    """Mapeia termos lidos por OCR em rótulos para materiais."""
    encontrados = []
    for tok in re.findall(r"[a-z]+", normalizar(texto_ocr)):
        tok = singularizar(tok)
        mat = CODIGOS_RECICLAGEM.get(tok)
        if mat and mat not in encontrados:
            encontrados.append(mat)
    return encontrados
