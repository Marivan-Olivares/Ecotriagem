"""Busca do ponto de coleta mais próximo em um grafo de bairros."""
import hashlib
import heapq
import json
import math
from collections import deque
from dataclasses import dataclass, field

from src import config


def haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def fator_tortuosidade(a: str, b: str) -> float:
    """Fator determinístico em [1.20, 1.60) — simula que ruas não são retas."""
    chave = "|".join(sorted((a, b))).encode()
    return 1.20 + 0.40 * (int(hashlib.md5(chave).hexdigest()[:8], 16) / 0xFFFFFFFF)


@dataclass
class ResultadoBusca:
    algoritmo: str
    caminho: list
    custo_km: float
    nos_expandidos: int
    destino: str | None = None
    pontos: list = field(default_factory=list)

    @property
    def encontrou(self):
        return bool(self.caminho)


class MapaColeta:
    def __init__(self, caminho=config.DIR_CONHECIMENTO / "mapa_maceio.json"):
        dados = json.load(open(caminho, encoding="utf-8"))
        self.coord = {b: tuple(c) for b, c in dados["bairros"].items()}
        self.adj = {b: {} for b in self.coord}
        for a, b in dados["vizinhos"]:
            custo = haversine_km(self.coord[a], self.coord[b]) * fator_tortuosidade(a, b)
            self.adj[a][b] = custo
            self.adj[b][a] = custo
        self.pontos = dados["pontos_coleta"]

    def objetivos(self, material: str | None) -> set:
        """Bairros com ponto que aceita o material (None = qualquer ponto)."""
        return {p["bairro"] for p in self.pontos if material is None or material in p["aceita"]}

    def pontos_no_bairro(self, bairro, material):
        return [p["nome"] for p in self.pontos
                if p["bairro"] == bairro and (material is None or material in p["aceita"])]

    def heuristica(self, no, objetivos) -> float:
        return min(haversine_km(self.coord[no], self.coord[g]) for g in objetivos)

    def custo_caminho(self, caminho) -> float:
        return sum(self.adj[a][b] for a, b in zip(caminho, caminho[1:]))

    def _resultado(self, nome, caminho, expandidos, material):
        if not caminho:
            return ResultadoBusca(nome, [], math.inf, expandidos)
        return ResultadoBusca(nome, caminho, self.custo_caminho(caminho), expandidos,
                              caminho[-1], self.pontos_no_bairro(caminho[-1], material))

    @staticmethod
    def _reconstruir(pais, fim):
        caminho = [fim]
        while pais[caminho[-1]] is not None:
            caminho.append(pais[caminho[-1]])
        return caminho[::-1]

    def bfs(self, inicio, material):
        objetivos = self.objetivos(material)
        pais, fila, expandidos = {inicio: None}, deque([inicio]), 0
        while fila:
            no = fila.popleft()
            expandidos += 1
            if no in objetivos:
                return self._resultado("BFS", self._reconstruir(pais, no), expandidos, material)
            for viz in sorted(self.adj[no]):
                if viz not in pais:
                    pais[viz] = no
                    fila.append(viz)
        return self._resultado("BFS", [], expandidos, material)

    def _melhor_primeiro(self, nome, inicio, material, f):
        """Esqueleto comum: UCS (f=g), Gulosa (f=h) e A* (f=g+h)."""
        objetivos = self.objetivos(material)
        if not objetivos:
            return self._resultado(nome, [], 0, material)
        g = {inicio: 0.0}
        pais = {inicio: None}
        fronteira = [(f(0.0, inicio, objetivos), 0, inicio)]
        fechados, expandidos, contador = set(), 0, 1
        while fronteira:
            _, _, no = heapq.heappop(fronteira)
            if no in fechados:
                continue
            fechados.add(no)
            expandidos += 1
            if no in objetivos:
                return self._resultado(nome, self._reconstruir(pais, no), expandidos, material)
            for viz, custo in self.adj[no].items():
                novo_g = g[no] + custo
                if viz not in fechados and novo_g < g.get(viz, math.inf):
                    g[viz] = novo_g
                    pais[viz] = no
                    heapq.heappush(fronteira, (f(novo_g, viz, objetivos), contador, viz))
                    contador += 1
        return self._resultado(nome, [], expandidos, material)

    def custo_uniforme(self, inicio, material):
        return self._melhor_primeiro("Custo uniforme", inicio, material, lambda g, n, o: g)

    def gulosa(self, inicio, material):
        return self._melhor_primeiro("Gulosa", inicio, material,
                                     lambda g, n, o: self.heuristica(n, o))

    def a_estrela(self, inicio, material):
        return self._melhor_primeiro("A*", inicio, material,
                                     lambda g, n, o: g + self.heuristica(n, o))
