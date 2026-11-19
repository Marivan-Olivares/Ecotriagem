"""Interface web do EcoTriagem."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.integracao.pipeline import EcoTriagem
from src.visao_computacional.preprocessamento import carregar_imagem

st.set_page_config(page_title="EcoTriagem", page_icon="♻️", layout="wide")


@st.cache_resource
def sistema():
    return EcoTriagem()


@st.cache_data
def exemplos_teste():
    df = pd.read_csv(config.ARQ_DIVISAO)
    t = df[df["particao"] == "teste"]
    return {f"{config.CLASSES_PT[c]} — {Path(p).name}": p
            for c, grupo in t.groupby("classe") for p in grupo["caminho"].head(4)}


def desenhar_rota(mapa, rota):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for a, viz in mapa.adj.items():
        for b in viz:
            ya, xa = mapa.coord[a]
            yb, xb = mapa.coord[b]
            ax.plot([xa, xb], [ya, yb], color="#cccccc", lw=0.8, zorder=1)
    for nome, (y, x) in mapa.coord.items():
        ax.scatter(x, y, s=10, color="#888888", zorder=2)
    for p in mapa.pontos:
        y, x = mapa.coord[p["bairro"]]
        ax.scatter(x, y, s=60, marker="s", color="#81b29a", zorder=3)
    if rota and rota["caminho"]:
        xs = [mapa.coord[b][1] for b in rota["caminho"]]
        ys = [mapa.coord[b][0] for b in rota["caminho"]]
        ax.plot(xs, ys, color="#e07a5f", lw=2.5, zorder=4)
        ax.annotate(rota["caminho"][0], (xs[0], ys[0]), fontsize=7)
        ax.annotate(rota["destino"], (xs[-1], ys[-1]), fontsize=7, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Bairros (cinza), pontos de coleta (verde), rota A* (laranja)", fontsize=7)
    return fig


eco = sistema()
if "conversa" not in st.session_state:
    st.session_state.conversa = []
    st.session_state.analise = None
    st.session_state.imagem_id = None

st.title("♻️ EcoTriagem")
st.caption("Identificação de resíduos por imagem + perguntas em linguagem natural + "
           "recomendação de descarte e rota até o ponto de coleta (Maceió — cenário simulado).")

col_img, col_chat = st.columns([1, 1.3])
with col_img:
    st.subheader("1. Imagem do resíduo")
    origem = st.radio("Origem", ["Enviar foto", "Exemplo do conjunto de teste"], horizontal=True)
    arquivo, chave = None, None
    if origem == "Enviar foto":
        up = st.file_uploader("Foto (jpg/png)", type=["jpg", "jpeg", "png"])
        if up is not None:
            arquivo, chave = up.getvalue(), f"up:{up.name}:{up.size}"
    else:
        ex = exemplos_teste()
        escolha = st.selectbox("Imagem", list(ex))
        arquivo, chave = config.RAIZ / ex[escolha], f"ex:{escolha}"
    if arquivo is not None and chave != st.session_state.imagem_id:
        eco.reiniciar()
        st.session_state.conversa = []
        st.session_state.analise = eco.analisar_imagem(arquivo)
        st.session_state.imagem_id = chave
    if arquivo is not None:
        img = carregar_imagem(arquivo)
        st.image(img[:, :, ::-1], width="stretch")
    a = st.session_state.analise
    if a:
        st.metric("Material previsto", a["material_pt"], f"confiança {a['confianca']:.0%}")
        probs = pd.DataFrame([(eco.kb[m]["nome"].split(" (")[0], p) for m, p in a["top"]],
                             columns=["material", "probabilidade"]).set_index("material")
        st.bar_chart(probs)
        st.write(f"**OCR do rótulo:** `{a['ocr_texto'] or '(nenhum texto legível)'}`"
                 + (f" → {', '.join(a['ocr_materiais'])}" if a["ocr_materiais"] else ""))
        st.caption(f"Tempo de análise: {a['tempo_total_ms']:.0f} ms "
                   f"(visão {a['tempo_visao_ms']:.0f} ms)")

with col_chat:
    st.subheader("2. Pergunte")
    for turno in st.session_state.conversa:
        with st.chat_message("user"):
            st.write(turno["pergunta"])
        with st.chat_message("assistant"):
            st.markdown(turno["resposta"])
            with st.expander("Como decidi"):
                st.write(f"Intenção: `{turno['intencao']}` ({turno['conf_intencao']:.2f}) · "
                         f"material: `{turno['material']}` · origem: {turno['origem_material']}")
                for r in turno["regras_disparadas"]:
                    st.write("• " + r)
            if turno.get("rota"):
                st.pyplot(desenhar_rota(eco.mapa, turno["rota"]))
    pergunta = st.chat_input("Ex.: onde descarto isso? / precisa lavar? / estou no Farol")
    if pergunta:
        resp = eco.perguntar(pergunta)
        st.session_state.conversa.append({"pergunta": pergunta, **resp})
        st.rerun()
    if st.button("Limpar conversa"):
        eco.reiniciar()
        st.session_state.conversa = []
        st.session_state.analise = None
        st.session_state.imagem_id = None
        st.rerun()
