"""Demonstração reproduzível do EcoTriagem pela linha de comando."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config
from src.integracao.pipeline import EcoTriagem

ROTEIRO = [
    ("plastic", 0, ["que material é esse?", "onde eu jogo isso?", "precisa tirar a tampa?",
                    "tem ponto de coleta perto?", "moro na Ponta Verde"]),
    ("glass", 1, ["isso é reciclável?", "quanto tempo leva pra se decompor?",
                  "onde fica o ponto de coleta mais próximo do Benedito Bentes?"]),
    ("cardboard", 2, ["qual lixeira uso pra isso?", "devo desmontar a caixa?"]),
    ("metal", 3, ["o que é isso na foto?", "dá pra reciclar?"]),
    ("trash", 0, ["onde descarto?", "tem ecoponto pra isso no Farol?"]),
    ("paper", 4, ["e se fosse uma pilha velha, onde levo? estou no Pinheiro"]),
    (None, None, ["oi", "o que é logística reversa?", "qual a previsão do tempo amanhã?",
                  "onde fica o ecoponto mais perto do Jacintinho?"]),
]


def imprimir(resp: dict):
    print(f"  intenção: {resp['intencao']} ({resp['conf_intencao']:.2f}) | material: "
          f"{resp['material']} | regras: {[r.split(':')[0] for r in resp['regras_disparadas']]}")
    for linha in resp["resposta"].split("\n"):
        if linha.strip():
            print("  << " + linha.replace("**", ""))


def roteiro():
    sistema = EcoTriagem()
    df = pd.read_csv(config.ARQ_DIVISAO)
    teste = df[df["particao"] == "teste"]
    registros, md = [], ["# Demonstração do EcoTriagem (gerada por `python demo.py --roteiro`)\n"]
    for classe, idx, perguntas in ROTEIRO:
        sistema.reiniciar()
        cena = {"imagem": None, "classe_real": classe, "dialogo": []}
        md.append("---\n")
        if classe:
            caminho = teste[teste["classe"] == classe]["caminho"].iloc[idx]
            analise = sistema.analisar_imagem(config.RAIZ / caminho)
            cena["imagem"] = caminho
            cena["analise"] = {k: analise[k] for k in ("classe", "material", "confianca",
                                                       "top", "ocr_texto", "ocr_materiais",
                                                       "tempo_total_ms")}
            print(f"\n[imagem] {caminho} (classe real: {classe})")
            print(f"  visão: {analise['material_pt']} ({analise['confianca']:.0%}) | OCR: "
                  f"'{analise['ocr_texto']}' -> {analise['ocr_materiais']}")
            md.append(f"**Imagem:** `{caminho}` — classe real: *{config.CLASSES_PT[classe]}*  \n"
                      f"**Visão:** {analise['material_pt']} ({analise['confianca']:.0%}); "
                      f"top-3: " + ", ".join(f"{m} {p:.0%}" for m, p in analise["top"][:3]) +
                      f"  \n**OCR:** `{analise['ocr_texto'] or '(nada legível)'}`\n")
        else:
            print("\n[sem imagem]")
            md.append("**Sem imagem** (apenas texto)\n")
        for p in perguntas:
            print(f"  >> {p}")
            r = sistema.perguntar(p)
            imprimir(r)
            cena["dialogo"].append({"pergunta": p, **r})
            regras = ", ".join(x.split(":")[0] for x in r["regras_disparadas"])
            md.append(f"- **Usuário:** {p}  \n  *intenção:* `{r['intencao']}` "
                      f"({r['conf_intencao']:.2f}); *regras:* {regras}  \n  **EcoTriagem:** "
                      + r["resposta"].replace("\n\n", " ") + "\n")
        registros.append(cena)
    config.DIR_EXEMPLOS.mkdir(parents=True, exist_ok=True)
    with open(config.DIR_EXEMPLOS / "demonstracao.json", "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2, default=float)
    (config.DIR_EXEMPLOS / "demonstracao.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsalvo em {config.DIR_EXEMPLOS.relative_to(config.RAIZ)}/demonstracao.(md|json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--roteiro", action="store_true")
    ap.add_argument("--imagem")
    ap.add_argument("--pergunta", action="append")
    a = ap.parse_args()
    if a.roteiro:
        return roteiro()
    sistema = EcoTriagem()
    if a.imagem:
        an = sistema.analisar_imagem(a.imagem)
        print(f"visão: {an['material_pt']} ({an['confianca']:.0%}) | OCR: '{an['ocr_texto']}'")
    perguntas = a.pergunta
    if perguntas:
        for p in perguntas:
            print(f">> {p}")
            imprimir(sistema.perguntar(p))
        return
    print("Digite perguntas (Enter vazio para sair).")
    while True:
        p = input(">> ").strip()
        if not p:
            break
        imprimir(sistema.perguntar(p))


if __name__ == "__main__":
    main()
