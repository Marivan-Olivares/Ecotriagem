"""Motor de inferência por encadeamento para frente (forward chaining)."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Regra:
    nome: str
    descricao: str
    prioridade: int
    condicao: Callable[[dict], bool]
    acao: Callable[[dict], None]


class MotorRegras:
    def __init__(self, regras: list[Regra], max_ciclos: int = 100):
        self.regras = sorted(regras, key=lambda r: r.prioridade)
        self.max_ciclos = max_ciclos

    def executar(self, fatos: dict) -> tuple[dict, list[str]]:
        disparadas, rastro = set(), []
        for _ in range(self.max_ciclos):
            aplicaveis = [r for r in self.regras
                          if r.nome not in disparadas and r.condicao(fatos)]
            if not aplicaveis:
                break
            regra = aplicaveis[0]
            regra.acao(fatos)
            disparadas.add(regra.nome)
            rastro.append(f"{regra.nome}: {regra.descricao}")
        return fatos, rastro
