"""Pré-processamento de texto em português."""
import re
import unicodedata

ABREVIACOES = {
    "vc": "voce", "vcs": "voces", "pra": "para", "pro": "para o", "q": "que",
    "tb": "tambem", "tbm": "tambem", "to": "estou", "ta": "esta", "pq": "porque",
    "n": "nao", "naum": "nao", "eh": "e", "refri": "refrigerante", "msm": "mesmo",
    "oq": "o que", "cmg": "comigo", "blz": "beleza", "obg": "obrigado", "vlw": "valeu",
}

STOPWORDS = set("""a o as os um uma uns umas de da do das dos em na no nas nos
por para com sem e ou ao aos isso isto esse essa este esta aquele aquela
me te se lhe eu voce ele ela nos eles elas meu minha seu sua
ja mais muito bem tambem so entao ai aqui la""".split())


def remover_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


def normalizar(texto: str) -> str:
    """Normaliza o texto mantendo palavras interrogativas."""
    t = remover_acentos(str(texto).lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = [ABREVIACOES.get(tok, tok) for tok in t.split()]
    t = " ".join(tokens)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenizar(texto: str) -> list[str]:
    return normalizar(texto).split()


def singularizar(token: str) -> str:
    """Singularização heurística para o português (suficiente para o léxico)."""
    if len(token) <= 3:
        return token
    for sufixo, troca in (("oes", "ao"), ("aes", "ao"), ("ais", "al"),
                          ("eis", "el"), ("ns", "m"), ("res", "r"), ("zes", "z")):
        if token.endswith(sufixo):
            return token[: -len(sufixo)] + troca
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens_singulares(texto: str) -> list[str]:
    return [singularizar(t) for t in tokenizar(texto)]
