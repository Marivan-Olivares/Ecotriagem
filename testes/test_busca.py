import itertools
import math

import pytest

from src.decisao.busca import MapaColeta, haversine_km

MATERIAIS = ["papel", "papelao", "plastico", "metal", "vidro", "perigoso", "oleo", "organico"]


@pytest.fixture(scope="module")
def mapa():
    return MapaColeta()


def test_grafo_conexo(mapa):
    inicio = next(iter(mapa.coord))
    visitados, pilha = {inicio}, [inicio]
    while pilha:
        for v in mapa.adj[pilha.pop()]:
            if v not in visitados:
                visitados.add(v)
                pilha.append(v)
    assert visitados == set(mapa.coord)


def test_custo_aresta_nunca_menor_que_linha_reta(mapa):
    for a, viz in mapa.adj.items():
        for b, c in viz.items():
            assert c >= haversine_km(mapa.coord[a], mapa.coord[b]) - 1e-9


def test_a_estrela_otimo_e_expande_menos_ou_igual(mapa):
    for inicio, mat in itertools.product(mapa.coord, MATERIAIS):
        a = mapa.a_estrela(inicio, mat)
        u = mapa.custo_uniforme(inicio, mat)
        assert math.isclose(a.custo_km, u.custo_km, rel_tol=1e-9)
        assert a.nos_expandidos <= u.nos_expandidos
        assert a.destino in mapa.objetivos(mat)


def test_heuristica_admissivel(mapa):
    for inicio, mat in itertools.product(mapa.coord, MATERIAIS):
        objetivos = mapa.objetivos(mat)
        assert mapa.heuristica(inicio, objetivos) <= mapa.custo_uniforme(inicio, mat).custo_km + 1e-9


def test_caminhos_validos(mapa):
    for alg in (mapa.bfs, mapa.gulosa, mapa.a_estrela):
        r = alg("Ipioca", "perigoso")
        assert r.caminho[0] == "Ipioca" and r.destino in mapa.objetivos("perigoso")
        for a, b in zip(r.caminho, r.caminho[1:]):
            assert b in mapa.adj[a]


def test_objetivo_no_proprio_bairro(mapa):
    r = mapa.a_estrela("Ponta Verde", "vidro")
    assert r.caminho == ["Ponta Verde"] and r.custo_km == 0
