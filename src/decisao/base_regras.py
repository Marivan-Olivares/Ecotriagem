"""Base de regras do EcoTriagem."""
from src import config
from src.decisao.regras import Regra

ALTA, MINIMA = config.LIMIAR_CONFIANCA_ALTA, config.LIMIAR_CONFIANCA_MINIMA
FORA_TRASHNET = {"perigoso", "oleo", "organico"}
PRECISA_MATERIAL = {"identificar_material", "como_descartar", "ponto_coleta",
                    "preparo_descarte", "reciclavel", "tempo_decomposicao"}


def _nome(f, m):
    return f["_kb"][m]["nome"]


def _conf(f):
    return f["visual"]["confianca"] if f.get("visual") else 0.0


def _top2(f):
    return " ou ".join(f"{_nome(f, m)} ({p:.0%})" for m, p in f["visual"]["top"][:2])


def _definir(f, material, origem):
    f["material"], f["origem_material"] = material, origem


def _km(x):
    return f"{x:.1f}".replace(".", ",")


def _diga(f, texto):
    f.setdefault("resposta", []).append(texto)


def _aviso(f, texto):
    f.setdefault("avisos", []).append(texto)


def r00(f):
    _definir(f, f["material_texto"], "texto da pergunta")


def r01(f):
    _aviso(f, f"Atenção: na imagem eu reconheci {_nome(f, f['visual']['material'])} "
              f"({_conf(f):.0%}), mas você mencionou {_nome(f, f['material'])}. "
              f"Vou responder sobre {_nome(f, f['material'])}.")


def r02(f):
    _definir(f, f["visual"]["material"], f"imagem ({_conf(f):.0%} de confiança)")


def r03(f):
    _definir(f, f["ocr_materiais"][0], "rótulo lido por OCR")
    _aviso(f, f"A imagem ficou ambígua ({_top2(f)}), mas li no rótulo um termo de "
              f"{_nome(f, f['material'])}; usei o rótulo.")


def r04(f):
    _definir(f, f["visual"]["material"], f"imagem ({_conf(f):.0%}, confiança moderada)")
    _aviso(f, f"Confiança moderada: parece {_top2(f)}. Confirme o material se puder.")


def r05(f):
    f["material"] = None
    f["pedir_foto"] = True
    _aviso(f, f"Não tenho segurança sobre o material (palpites: {_top2(f)}). "
              "Envie outra foto, com o objeto centralizado e boa iluminação, "
              "ou diga qual é o objeto.")


def r06(f):
    f["material"] = None
    f["pedir_foto"] = True
    _aviso(f, "Para responder preciso saber o material: envie uma foto do resíduo "
              "ou diga qual é o objeto (ex.: 'garrafa PET', 'latinha').")


def r07(f):
    _aviso(f, f"O rótulo menciona {_nome(f, f['ocr_materiais'][0])}, mas a imagem indica "
              f"{_nome(f, f['material'])} com alta confiança; mantive a imagem.")


def r10(f):
    f["info"] = f["_kb"][f["material"]]


def r20(f):
    i = f["info"]
    extra = ""
    if f.get("visual") and f["origem_material"].startswith("imagem"):
        extra = " Probabilidades: " + ", ".join(
            f"{_nome(f, m)} {p:.0%}" for m, p in f["visual"]["top"][:3]) + "."
    _diga(f, f"O material é **{i['nome']}** (fonte: {f['origem_material']}).{extra}")


def r21(f):
    i = f["info"]
    if i["reciclavel"] and i["categoria"] == "reciclável":
        _diga(f, f"Descarte **{i['nome']}** na lixeira **{i['lixeira']}** da coleta seletiva.")
    elif f["material"] in ("perigoso", "oleo"):
        _diga(f, f"**{i['nome'].capitalize()}** não vai no lixo comum: leve a um ponto de "
                 f"coleta específico ({i['categoria']}). Pergunte o seu bairro e eu "
                 "indico o ponto mais próximo.")
    else:
        _diga(f, f"**{i['nome'].capitalize()}** vai na lixeira **{i['lixeira']}** "
                 "e segue pela coleta regular, separado dos recicláveis.")


def r22(f):
    i = f["info"]
    if i["reciclavel"] and i["categoria"] == "reciclável":
        _diga(f, f"Sim, **{i['nome']}** é reciclável: separe para a coleta seletiva "
                 f"(lixeira {i['lixeira']}).")
    elif i["reciclavel"]:
        _diga(f, f"Sim, **{i['nome']}** pode ser reaproveitado, mas por {i['categoria']} "
                 "(ponto de coleta específico), não pela coleta seletiva comum.")
    else:
        _diga(f, f"Não: **{i['nome']}** não é reciclável na coleta seletiva comum "
                 f"({i['categoria']}).")


def r23(f):
    i = f["info"]
    _diga(f, f"Antes de descartar {i['nome']}: " + "; ".join(i["preparo"]) + ".")


def r24(f):
    i = f["info"]
    _diga(f, f"Tempo de decomposição estimado de {i['nome']}: {i['decomposicao']}. "
             "(Valores aproximados; dependem muito do ambiente.)")


def r25_sem_bairro(f):
    _diga(f, "Em qual bairro de Maceió você está? Assim calculo a rota até o ponto "
             "de coleta mais próximo que aceita esse material.")


def r25_rejeito(f):
    _diga(f, f"{_nome(f, f['material']).capitalize()} não precisa ir a ecoponto: "
             "entregue na coleta regular de lixo, em saco fechado.")


def r25_rota(f):
    rota = f["_mapa"].a_estrela(f["bairro"], f["material"])
    f["rota"] = rota
    if not rota.encontrou:
        _diga(f, f"Não encontrei ponto que aceite {_nome(f, f['material'])} no mapa.")
        return
    pontos = ", ".join(rota.pontos)
    if len(rota.caminho) == 1:
        _diga(f, f"Há ponto de coleta no seu próprio bairro ({f['bairro']}): {pontos}.")
    else:
        _diga(f, f"Ponto de coleta mais próximo para {_nome(f, f['material'])} saindo de "
                 f"{f['bairro']}: **{pontos}** ({rota.destino}), cerca de "
                 f"{_km(rota.custo_km)} km. Trajeto: {' → '.join(rota.caminho)}. "
                 f"(Busca A*, {rota.nos_expandidos} bairros expandidos.)")
    _diga(f, "Obs.: pontos de coleta do mapa são fictícios (cenário de demonstração).")


def r25_qualquer(f):
    rota = f["_mapa"].a_estrela(f["bairro"], None)
    f["rota"] = rota
    if len(rota.caminho) == 1:
        _diga(f, f"Há ponto de coleta no seu próprio bairro ({f['bairro']}): "
                 f"**{', '.join(rota.pontos)}**.")
    else:
        _diga(f, f"Ponto de coleta mais próximo de {f['bairro']}: **{', '.join(rota.pontos)}** "
                 f"({rota.destino}), cerca de {_km(rota.custo_km)} km. Trajeto: "
                 f"{' → '.join(rota.caminho)}.")
    _diga(f, "Envie a foto do resíduo para eu conferir se o ponto aceita o material.")
    _diga(f, "Obs.: pontos de coleta do mapa são fictícios (cenário de demonstração).")


def r26(f):
    resp, sim, _ = f["_faq"].responder(f["texto"])
    if resp:
        _diga(f, resp)
    else:
        _diga(f, "Não encontrei essa informação na minha base. Posso ajudar a identificar "
                 "um resíduo, indicar a lixeira correta ou o ponto de coleta mais próximo.")


def r27(f):
    _diga(f, "Olá! Sou o EcoTriagem. Envie a foto de um resíduo e pergunte, por exemplo: "
             "'que material é esse?', 'onde descarto?', 'precisa lavar?', 'é reciclável?' "
             "ou 'qual o ponto de coleta mais perto do Farol?'.")


def r28(f):
    _diga(f, "Desculpe, só consigo ajudar com identificação e descarte de resíduos.")


def r29(f):
    opcoes = ", ".join(i.replace("_", " ") for i, _ in f["ranking_intencao"][:2])
    _diga(f, f"Não entendi bem a pergunta (talvez: {opcoes}). Pode reformular?")


def _tem_imagem(f):
    return f.get("visual") is not None


def _intencao(*nomes):
    return lambda f: (f.get("intencao") in nomes and not f.get("intencao_incerta")
                      and f.get("material"))


REGRAS = [
    Regra("R00", "material citado no texto prevalece (referência explícita do usuário)", 0,
          lambda f: bool(f.get("material_texto")) and "material" not in f, r00),
    Regra("R01", "conflito entre texto e imagem de alta confiança -> avisar", 1,
          lambda f: f.get("origem_material") == "texto da pergunta" and _tem_imagem(f)
          and _conf(f) >= ALTA and f["visual"]["material"] != f["material"], r01),
    Regra("R02", f"imagem com confiança >= {ALTA:.0%} -> aceitar classe visual", 2,
          lambda f: "material" not in f and _tem_imagem(f) and _conf(f) >= ALTA, r02),
    Regra("R03", "imagem ambígua + rótulo lido por OCR -> usar rótulo (fusão)", 3,
          lambda f: "material" not in f and _tem_imagem(f) and _conf(f) < ALTA
          and bool(f.get("ocr_materiais")), r03),
    Regra("R04", f"confiança entre {MINIMA:.0%} e {ALTA:.0%} -> aceitar e pedir confirmação", 4,
          lambda f: "material" not in f and _tem_imagem(f) and MINIMA <= _conf(f) < ALTA, r04),
    Regra("R05", f"confiança < {MINIMA:.0%} -> não decidir; pedir nova foto", 5,
          lambda f: "material" not in f and _tem_imagem(f) and _conf(f) < MINIMA, r05),
    Regra("R06", "pergunta exige material mas não há imagem nem menção -> pedir foto", 6,
          lambda f: "material" not in f and not _tem_imagem(f)
          and f.get("intencao") in PRECISA_MATERIAL and not f.get("intencao_incerta")
          and not (f.get("intencao") == "ponto_coleta" and f.get("bairro")), r06),
    Regra("R07", "rótulo OCR contradiz imagem de alta confiança -> avisar", 7,
          lambda f: bool(f.get("ocr_materiais")) and f.get("origem_material", "").startswith("imagem")
          and _conf(f) >= ALTA and f["ocr_materiais"][0] != f["material"], r07),
    Regra("R10", "consultar a base de conhecimento do material", 10,
          lambda f: bool(f.get("material")) and "info" not in f, r10),
    Regra("R29", "intenção com baixa confiança -> pedir reformulação", 19,
          lambda f: f.get("intencao_incerta", False), r29),
    Regra("R20", "intenção identificar_material -> informar material e probabilidades", 20,
          _intencao("identificar_material"), r20),
    Regra("R21", "intenção como_descartar -> lixeira (cores CONAMA 275/2001)", 21,
          _intencao("como_descartar"), r21),
    Regra("R22", "intenção reciclavel -> reciclável ou não", 22, _intencao("reciclavel"), r22),
    Regra("R23", "intenção preparo_descarte -> cuidados antes do descarte", 23,
          _intencao("preparo_descarte"), r23),
    Regra("R24", "intenção tempo_decomposicao -> impacto/decomposição", 24,
          _intencao("tempo_decomposicao"), r24),
    Regra("R25a", "ponto_coleta sem bairro conhecido -> perguntar bairro", 25,
          lambda f: _intencao("ponto_coleta")(f) and not f.get("bairro")
          and f["material"] not in ("rejeito",), r25_sem_bairro),
    Regra("R25b", "ponto_coleta para rejeito -> coleta regular (sem busca)", 25,
          lambda f: _intencao("ponto_coleta")(f) and f["material"] == "rejeito", r25_rejeito),
    Regra("R25c", "ponto_coleta com bairro -> busca A* até ponto que aceita o material", 25,
          lambda f: _intencao("ponto_coleta")(f) and bool(f.get("bairro"))
          and f["material"] != "rejeito", r25_rota),
    Regra("R25d", "ponto_coleta com bairro e sem material -> A* até qualquer ponto", 25,
          lambda f: f.get("intencao") == "ponto_coleta" and not f.get("intencao_incerta")
          and not f.get("material") and not _tem_imagem(f) and bool(f.get("bairro")),
          r25_qualquer),
    Regra("R26", "intenção duvida_geral -> perguntas e respostas (busca na FAQ)", 26,
          lambda f: f.get("intencao") == "duvida_geral" and not f.get("intencao_incerta"), r26),
    Regra("R27", "intenção saudacao_ajuda -> apresentar o assistente", 27,
          lambda f: f.get("intencao") == "saudacao_ajuda" and not f.get("intencao_incerta"), r27),
    Regra("R28", "intenção fora_escopo -> recusar educadamente", 28,
          lambda f: f.get("intencao") == "fora_escopo" and not f.get("intencao_incerta"), r28),
]
