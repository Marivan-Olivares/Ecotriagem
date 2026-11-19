import json

import pytest

from src import config
from src.decisao.base_regras import REGRAS
from src.decisao.busca import MapaColeta
from src.decisao.regras import MotorRegras, Regra
from src.pln.faq import BuscaFAQ


@pytest.fixture(scope="module")
def servicos():
    kb = {k: v for k, v in json.load(open(config.DIR_CONHECIMENTO / "materiais.json",
                                          encoding="utf-8")).items() if not k.startswith("_")}
    return {"_kb": kb, "_mapa": MapaColeta(), "_faq": BuscaFAQ()}


def rodar(servicos, **fatos):
    base = {"texto": "", "intencao": "como_descartar", "intencao_incerta": False,
            "ranking_intencao": [("como_descartar", 0.9)], "material_texto": None,
            "bairro": None, "visual": None, "ocr_materiais": []}
    base.update(fatos)
    base.update(servicos)
    f, rastro = MotorRegras(REGRAS).executar(base)
    return f, [r.split(":")[0] for r in rastro]


ALTA, MIN = config.LIMIAR_CONFIANCA_ALTA, config.LIMIAR_CONFIANCA_MINIMA
MODERADA = (ALTA + MIN) / 2
BAIXA = MIN - 0.1


def vis(material, conf, segundo="plastico"):
    return {"material": material, "confianca": conf,
            "top": [(material, conf), (segundo, 1 - conf)]}


def test_motor_ponto_fixo_e_prioridade():
    ordem = []
    regras = [Regra("B", "", 2, lambda f: "a" in f, lambda f: ordem.append("B")),
              Regra("A", "", 1, lambda f: True, lambda f: f.update(a=1) or ordem.append("A"))]
    MotorRegras(regras).executar({})
    assert ordem == ["A", "B"]


def test_alta_confianca_aceita_imagem(servicos):
    f, r = rodar(servicos, visual=vis("vidro", 0.95))
    assert f["material"] == "vidro" and "R02" in r and "R21" in r
    assert "verde" in " ".join(f["resposta"])


def test_baixa_confianca_com_ocr_usa_rotulo(servicos):
    f, r = rodar(servicos, visual=vis("vidro", MODERADA), ocr_materiais=["plastico"])
    assert f["material"] == "plastico" and "R03" in r


def test_confianca_moderada_pede_confirmacao(servicos):
    f, r = rodar(servicos, visual=vis("metal", MODERADA))
    assert f["material"] == "metal" and "R04" in r and f["avisos"]


def test_confianca_baixa_nao_decide(servicos):
    f, r = rodar(servicos, visual=vis("metal", BAIXA))
    assert f["material"] is None and "R05" in r and "R21" not in r


def test_texto_prevalece_e_avisa_conflito(servicos):
    f, r = rodar(servicos, visual=vis("vidro", 0.95), material_texto="metal")
    assert f["material"] == "metal" and r[:2] == ["R00", "R01"]


def test_sem_imagem_e_sem_material_pede_foto(servicos):
    f, r = rodar(servicos)
    assert "R06" in r and f["material"] is None


def test_ponto_coleta_dispara_a_estrela(servicos):
    f, r = rodar(servicos, intencao="ponto_coleta", visual=vis("vidro", 0.95), bairro="Farol")
    assert "R25c" in r and f["rota"].destino in servicos["_mapa"].objetivos("vidro")


def test_rejeito_nao_busca_rota(servicos):
    f, r = rodar(servicos, intencao="ponto_coleta", visual=vis("rejeito", 0.95), bairro="Farol")
    assert "R25b" in r and "rota" not in f


def test_intencao_incerta(servicos):
    f, r = rodar(servicos, intencao_incerta=True, visual=vis("vidro", 0.95))
    assert "R29" in r and "R21" not in r
