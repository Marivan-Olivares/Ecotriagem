import pytest

from src import config
from src.pln.entidades import ExtratorEntidades, materiais_em_rotulo
from src.pln.faq import BuscaFAQ
from src.pln.preprocessamento import normalizar, singularizar


def test_normalizar_remove_acentos_e_expande_girias():
    assert normalizar("Vc sabe ONDE eu jogo isso, pra reciclar??") == \
        "voce sabe onde eu jogo isso para reciclar"


@pytest.mark.parametrize("plural,singular", [("garrafas", "garrafa"), ("latões", "latoes"),
                                              ("pilhas", "pilha"), ("papéis", "papéis")])
def test_singularizar(plural, singular):
    assert singularizar(normalizar(plural)) == normalizar(singular).replace("latoes", "latao") \
        .replace("papeis", "papel")


@pytest.fixture(scope="module")
def ext():
    return ExtratorEntidades()


@pytest.mark.parametrize("texto,material", [
    ("onde jogo a latinha de refri?", "metal"),
    ("papel higiênico usado é reciclável?", "rejeito"),
    ("tenho umas garrafas de vidro aqui", "vidro"),
    ("descarte de garafa pet", "plastico"),
    ("onde levo pilhas velhas", "perigoso"),
    ("que material é esse da foto?", None),
])
def test_extrator_material(ext, texto, material):
    assert ext.extrair(texto).material == material


@pytest.mark.parametrize("texto,bairro", [
    ("moro na Ponta Verde", "Ponta Verde"), ("estou no tabuleiro", "Tabuleiro do Martins"),
    ("perto da jatiuca", "Jatiúca"), ("qual a capital da França", None)])
def test_extrator_bairro(ext, texto, bairro):
    assert ext.extrair_bairro(texto) == bairro


def test_ocr_rotulo_para_material():
    assert materiais_em_rotulo("Garrafa PET 1 reciclável") == ["plastico"]
    assert materiais_em_rotulo("LATA DE ALUMÍNIO") == ["metal"]


def test_faq_recupera_e_abstem():
    faq = BuscaFAQ()
    resp, sim, perg = faq.responder("o que significa logística reversa?")
    assert perg == "O que é logística reversa?" and resp
    resp, _, _ = faq.responder("qual a capital da Argentina")
    assert resp is None


@pytest.mark.skipif(not config.ARQ_MODELO_INTENCAO.exists(), reason="treine o modelo de PLN")
@pytest.mark.parametrize("texto,intencao", [
    ("onde eu jogo essa garrafa?", "como_descartar"),
    ("isso é reciclável?", "reciclavel"),
    ("tem ecoponto perto do Farol?", "ponto_coleta"),
    ("preciso lavar antes?", "preparo_descarte"),
    ("bom dia", "saudacao_ajuda"),
])
def test_intencoes_basicas(texto, intencao):
    from src.pln.intencoes import ClassificadorIntencao
    assert ClassificadorIntencao().prever(texto).intencao == intencao
