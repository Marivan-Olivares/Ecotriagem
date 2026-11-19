import numpy as np
import pandas as pd

from src import config
from src.visao_computacional import preprocessamento as pp
from src.visao_computacional.extrator_classico import caracteristicas_classicas
from testes.conftest import precisa_modelos


def img_teste(classe="glass", i=0):
    df = pd.read_csv(config.ARQ_DIVISAO)
    return config.RAIZ / df[(df.particao == "teste") & (df.classe == classe)].caminho.iloc[i]


def test_perturbacoes_mantem_formato():
    img = np.random.default_rng(0).integers(0, 255, (384, 512, 3), dtype=np.uint8)
    for nome, f in pp.PERTURBACOES.items():
        assert f(img).shape == img.shape, nome


def test_caracteristicas_classicas_dimensao():
    img = np.full((384, 512, 3), 128, np.uint8)
    assert caracteristicas_classicas(img).shape == (1423,)


@precisa_modelos
def test_extrator_cnn_deterministico():
    from src.visao_computacional.extrator_cnn import ExtratorMobileNet
    img = pp.carregar_imagem(img_teste())
    e = ExtratorMobileNet()
    a, b = e.extrair(img), e.extrair(img)
    assert a.shape == (1, 1280) and np.allclose(a, b)


@precisa_modelos
def test_classificador_probabilidades():
    from src.visao_computacional.classificador import ClassificadorVisual
    r = ClassificadorVisual().classificar(pp.carregar_imagem(img_teste("cardboard")))
    assert abs(sum(r.probabilidades.values()) - 1) < 1e-6
    assert r.classe in config.CLASSES and 0 <= r.confianca <= 1


@precisa_modelos
def test_fluxo_integrado_com_memoria():
    from src.integracao.pipeline import EcoTriagem
    eco = EcoTriagem()
    analise = eco.analisar_imagem(img_teste("cardboard", 2))
    assert analise["material"] in ("papelao", "papel", "plastico", "vidro", "metal", "rejeito")
    r1 = eco.perguntar("onde descarto isso?")
    assert r1["intencao"] == "como_descartar" and r1["material"] == analise["material"]
    r2 = eco.perguntar("qual o ponto de coleta mais perto?")
    assert r2["intencao"] == "ponto_coleta" and "bairro" in r2["resposta"]
    r3 = eco.perguntar("estou no Farol")
    assert r3["rota"] is not None or analise["material"] == "rejeito"
