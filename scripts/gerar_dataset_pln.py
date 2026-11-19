"""Gera o conjunto de TREINO de intenções (dataset autoral, em português)."""
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

AMOSTRAS_POR_INTENCAO = 130

TEMPLATES = {
    "identificar_material": [
        "o que é isso", "que material é esse", "isso é feito de quê",
        "do que é feito esse objeto", "você consegue identificar o material da foto",
        "qual é o material desse {obj}", "me diz que tipo de resíduo é esse",
        "isso aqui é plástico ou vidro", "esse {obj} é de metal",
        "identifica pra mim o que tem na imagem", "que tipo de lixo é esse",
        "classifica essa foto", "qual a classe desse objeto",
        "o que você vê na imagem", "reconhece esse material pra mim",
        "isso é papel ou papelão", "de que material é essa embalagem",
        "analisa a foto e me diz o material", "consegue reconhecer esse {obj}",
        "que resíduo aparece na foto", "o que tem nessa imagem que enviei",
        "qual material você detectou", "esse objeto é reciclável ou rejeito, qual o material",
    ],
    "como_descartar": [
        "onde eu jogo isso", "em qual lixeira coloco esse {obj}",
        "como descarto {obj}", "qual a cor da lixeira para {obj}",
        "onde devo descartar isso", "em que lixo vai esse {obj}",
        "como faço o descarte correto disso", "qual lixeira usar",
        "jogo {obj} no lixo comum", "posso colocar {obj} na lixeira azul",
        "qual o destino certo pra esse {obj}", "como me livro desse {obj}",
        "onde coloco {obj} usado", "{obj} vai em qual lixeira",
        "qual a forma correta de descartar {obj}", "me ajuda a descartar isso",
        "o que faço com esse {obj} depois de usar", "como separar {obj} pra coleta",
        "isso vai no reciclável ou no lixo normal", "qual coletor recebe {obj}",
        "descarte de {obj} como funciona", "posso jogar {obj} junto com o resto do lixo",
    ],
    "ponto_coleta": [
        "onde fica o ecoponto mais próximo", "tem ponto de coleta perto do {bairro}",
        "estou no {bairro}, onde levo {obj}", "qual o ponto de entrega mais perto de mim",
        "onde posso levar {obj} aqui no {bairro}", "me mostra o caminho até o ponto de coleta",
        "qual ecoponto aceita {obj}", "moro no {bairro}, onde entrego isso",
        "tem cooperativa de reciclagem perto do {bairro}", "rota até o ponto de coleta mais próximo",
        "onde descartar {obj} perto do {bairro}", "qual o local de coleta mais próximo do {bairro}",
        "existe PEV no {bairro}", "onde tem coleta de {obj} em Maceió",
        "preciso de um lugar pra entregar {obj}", "como chego no ecoponto saindo do {bairro}",
        "qual a distância até o ponto de coleta", "procuro um ponto que receba {obj}",
        "indica um local de entrega voluntária perto do {bairro}", "endereço do ponto de coleta mais perto",
        "onde fica a reciclagem mais perto da minha casa no {bairro}",
    ],
    "preparo_descarte": [
        "preciso lavar antes de descartar", "tenho que tirar a tampa do {obj}",
        "precisa limpar o {obj} antes de jogar fora", "devo amassar o {obj}",
        "como preparar o {obj} para reciclagem", "tem que tirar o rótulo",
        "preciso separar a tampa", "precisa enxaguar a embalagem",
        "o {obj} tem que estar seco", "devo desmontar a caixa",
        "o que fazer antes de descartar {obj}", "precisa retirar o resto de comida do {obj}",
        "posso descartar {obj} sujo", "como embalar {obj} quebrado",
        "tem algum cuidado antes de jogar {obj} fora", "preciso tirar a fita adesiva da caixa",
        "devo dobrar o {obj}", "tem que lavar {obj} com sabão",
        "como deixar o {obj} pronto pra coleta seletiva", "qual cuidado devo ter com {obj} antes do descarte",
    ],
    "reciclavel": [
        "isso é reciclável", "dá pra reciclar esse {obj}", "{obj} é reciclável",
        "esse material pode ser reciclado", "{obj} recicla", "tem como reciclar isso",
        "{obj} vai pra reciclagem", "o {obj} é aproveitado na reciclagem",
        "esse {obj} pode ser reaproveitado", "reciclam {obj} aqui",
        "isso tem valor pra reciclagem", "{obj} é lixo reciclável ou não",
        "posso mandar {obj} pra reciclar", "a cooperativa aceita {obj}",
        "esse objeto da foto é reciclável", "o material dessa imagem recicla",
        "{obj} sujo ainda é reciclável", "é possível reciclar {obj}",
        "{obj} entra na coleta seletiva", "vale a pena separar {obj} para reciclar",
    ],
    "tempo_decomposicao": [
        "quanto tempo isso leva pra se decompor", "quanto tempo o {obj} demora pra decompor",
        "qual o tempo de decomposição do {obj}", "{obj} demora quanto tempo na natureza",
        "em quantos anos o {obj} se decompõe", "isso some da natureza em quanto tempo",
        "qual o impacto ambiental de jogar {obj} no chão", "quanto tempo dura {obj} no meio ambiente",
        "{obj} se degrada rápido", "o {obj} polui por quanto tempo",
        "jogar {obj} na praia faz mal", "qual o dano ambiental do {obj}",
        "o {obj} é biodegradável", "quanto tempo esse material leva pra desaparecer",
        "qual a durabilidade do {obj} no aterro", "isso prejudica o meio ambiente",
        "quanto tempo leva pra decompor o material da foto", "{obj} contamina o solo",
    ],
    "duvida_geral": [
        "o que é coleta seletiva", "o que significa logística reversa",
        "o que fazer com remédio vencido", "qual a diferença entre rejeito e resíduo",
        "quais são as cores das lixeiras", "por que devo separar o lixo",
        "o que é resíduo perigoso", "por que reciclar é importante",
        "qual a diferença entre lixão e aterro", "o que fazer com entulho de obra",
        "como fazer compostagem em casa", "o que são cooperativas de reciclagem",
        "como os catadores trabalham", "lixo hospitalar pode ir no lixo comum",
        "como funciona esse aplicativo", "o que é a política nacional de resíduos sólidos",
        "como funciona a reciclagem", "explica o que é compostagem",
        "qual a importância dos catadores", "pra que serve a lixeira laranja",
        "o que é PEV", "o que é ecoponto",
    ],
    "saudacao_ajuda": [
        "oi", "olá", "bom dia", "boa tarde", "boa noite", "oi tudo bem",
        "olá, preciso de ajuda", "como você funciona", "o que você sabe fazer",
        "ajuda", "me ajuda", "quais perguntas posso fazer", "e aí",
        "olá assistente", "obrigado", "valeu", "muito obrigada pela ajuda",
        "tchau", "até mais", "você pode me ajudar", "quem é você", "começar",
    ],
    "fora_escopo": [
        "qual a previsão do tempo amanhã", "quem ganhou o jogo ontem",
        "me conta uma piada", "qual a capital da França", "quanto é 2 mais 2",
        "me recomenda um filme", "qual o melhor restaurante de Maceió",
        "como faço bolo de chocolate", "que horas são", "qual o preço do dólar",
        "toca uma música", "escreve um poema", "qual o resultado da loteria",
        "como resolver equação do segundo grau", "me ajuda com meu trabalho de história",
        "qual o horário do ônibus", "quem descobriu o Brasil",
        "como está o trânsito na Fernandes Lima", "qual a melhor praia de Alagoas",
        "como aprender inglês rápido", "me empresta dinheiro", "você gosta de futebol",
    ],
}

PREFIXOS = ["", "", "", "", "oi, ", "por favor, ", "me diz, ", "olha, ",
            "uma dúvida: ", "pergunta: ", "ei, ", "bom dia, "]
SUFIXOS = ["", "", "", "", "?", "?", "??", " por favor", " ?", "!"]
GIRIAS = [("você", "vc"), ("para", "pra"), ("que", "q"), ("está", "tá"),
          ("estou", "tô"), ("também", "tb"), ("porque", "pq")]


def sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t)
                   if unicodedata.category(c) != "Mn")


def erro_digitacao(t: str, rng: random.Random) -> str:
    if len(t) < 6:
        return t
    i = rng.randrange(1, len(t) - 2)
    op = rng.choice(["troca", "apaga", "duplica"])
    if op == "troca":
        return t[:i] + t[i + 1] + t[i] + t[i + 2:]
    if op == "apaga":
        return t[:i] + t[i + 1:]
    return t[:i] + t[i] + t[i:]


def aumentar(t: str, rng: random.Random) -> str:
    t = rng.choice(PREFIXOS) + t + rng.choice(SUFIXOS)
    for formal, girias in GIRIAS:
        if rng.random() < 0.35:
            t = re.sub(rf"\b{formal}\b", girias, t)
    if rng.random() < 0.30:
        t = sem_acento(t)
    if rng.random() < 0.20:
        t = erro_digitacao(t, rng)
    if rng.random() < 0.15:
        t = t.capitalize()
    return t


def main() -> None:
    rng = random.Random(config.SEMENTE)
    objetos = json.load(open(config.DIR_CONHECIMENTO / "objetos.json", encoding="utf-8"))
    termos = [(t, k) for k, v in objetos.items() if not k.startswith("_") for t in v]
    termos += [(g, "") for g in ["isso", "esse objeto", "esse material", "essa embalagem"]]
    bairros = list(json.load(open(config.DIR_CONHECIMENTO / "mapa_maceio.json",
                                  encoding="utf-8"))["bairros"])
    linhas = []
    for intencao, modelos in TEMPLATES.items():
        vistos = set()
        tentativas = 0
        while len(vistos) < AMOSTRAS_POR_INTENCAO and tentativas < 20000:
            tentativas += 1
            modelo = rng.choice(modelos)
            termo, material = rng.choice(termos)
            bairro = rng.choice(bairros)
            frase = aumentar(modelo.format(obj=termo, bairro=bairro), rng)
            if frase not in vistos:
                vistos.add(frase)
                linhas.append({"texto": frase, "intencao": intencao,
                               "material": material if "{obj}" in modelo else "",
                               "bairro": bairro if "{bairro}" in modelo else ""})
    df = pd.DataFrame(linhas).sample(frac=1, random_state=config.SEMENTE)
    config.DIR_PLN.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DIR_PLN / "intencoes_treino.csv", index=False)
    print(df["intencao"].value_counts())
    print(f"total: {len(df)} frases | templates: "
          f"{sum(len(v) for v in TEMPLATES.values())}")


if __name__ == "__main__":
    main()
