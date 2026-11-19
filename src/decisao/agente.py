"""Agente inteligente baseado em objetivos (Russell & Norvig)."""
from dataclasses import dataclass, field

from src.decisao.base_regras import REGRAS
from src.decisao.regras import MotorRegras
from src.pln.intencoes import LIMIAR_INTENCAO


@dataclass
class EstadoAgente:
    visual: dict | None = None
    ocr_texto: str = ""
    ocr_materiais: list = field(default_factory=list)
    bairro: str | None = None
    ultimo_material_texto: str | None = None
    ultima_intencao: str | None = None
    historico: list = field(default_factory=list)
    avisos_emitidos: set = field(default_factory=set)


class AgenteEcoTriagem:
    def __init__(self, classificador_intencao, extrator_entidades, kb, mapa, faq):
        self.intencao = classificador_intencao
        self.entidades = extrator_entidades
        self.servicos = {"_kb": kb, "_mapa": mapa, "_faq": faq}
        self.motor = MotorRegras(REGRAS)
        self.estado = EstadoAgente()

    def reiniciar(self):
        self.estado = EstadoAgente()

    def perceber_imagem(self, visual: dict, ocr_texto: str, ocr_materiais: list):
        self.estado.visual = visual
        self.estado.ocr_texto = ocr_texto
        self.estado.ocr_materiais = ocr_materiais
        self.estado.ultimo_material_texto = None
        self.estado.avisos_emitidos = set()

    def perceber_texto(self, texto: str) -> dict:
        r = self.intencao.prever(texto)
        ent = self.entidades.extrair(texto)
        intencao, incerta = r.intencao, r.incerta
        if (ent.bairro and self.estado.ultima_intencao == "ponto_coleta"
                and len(texto.split()) <= 5 and r.intencao != "ponto_coleta"):
            intencao, incerta = "ponto_coleta", False
        if ent.bairro:
            self.estado.bairro = ent.bairro
        if ent.material:
            self.estado.ultimo_material_texto = ent.material
        return {"texto": texto, "intencao": intencao, "conf_intencao": r.confianca,
                "intencao_incerta": incerta, "ranking_intencao": r.ranking,
                "material_texto": ent.material, "bairro": self.estado.bairro,
                "entidades_material": [m for m, _, _ in ent.materiais]}

    def decidir(self, percepcao: dict) -> dict:
        fatos = dict(percepcao)
        fatos.update(self.servicos)
        if not fatos.get("material_texto") and self.estado.ultimo_material_texto \
                and not self.estado.visual:
            fatos["material_texto"] = self.estado.ultimo_material_texto
        fatos["visual"] = self.estado.visual
        fatos["ocr_materiais"] = self.estado.ocr_materiais
        fatos, rastro = self.motor.executar(fatos)
        self.estado.ultima_intencao = fatos.get("intencao")
        avisos = [a for a in fatos.get("avisos", []) if a not in self.estado.avisos_emitidos]
        self.estado.avisos_emitidos.update(avisos)
        saida = {
            "resposta": "\n\n".join(avisos + fatos.get("resposta", [])),
            "intencao": fatos.get("intencao"), "conf_intencao": fatos.get("conf_intencao"),
            "material": fatos.get("material"), "origem_material": fatos.get("origem_material"),
            "bairro": fatos.get("bairro"), "regras_disparadas": rastro,
            "rota": None,
        }
        if fatos.get("rota") is not None:
            r = fatos["rota"]
            saida["rota"] = {"caminho": r.caminho, "custo_km": round(r.custo_km, 2),
                             "destino": r.destino, "pontos": r.pontos,
                             "nos_expandidos": r.nos_expandidos}
        self.estado.historico.append({"pergunta": percepcao["texto"], **saida})
        return saida

    def responder(self, texto: str) -> dict:
        return self.decidir(self.perceber_texto(texto))


__all__ = ["AgenteEcoTriagem", "LIMIAR_INTENCAO"]
