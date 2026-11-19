"""Configurações centrais do projeto EcoTriagem."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DIR_DADOS = RAIZ / "dados"
DIR_BRUTO = DIR_DADOS / "bruto"
DIR_IMAGENS = DIR_BRUTO / "dataset-resized"
DIR_PROCESSADO = DIR_DADOS / "processado"
ARQ_DIVISAO = DIR_PROCESSADO / "divisao.csv"
DIR_PLN = DIR_DADOS / "pln"
DIR_CONHECIMENTO = DIR_DADOS / "conhecimento"

DIR_MODELOS = RAIZ / "modelos"
ARQ_ONNX = DIR_MODELOS / "mobilenetv2-12.onnx"
ARQ_MODELO_VISAO = DIR_MODELOS / "classificador_visual.joblib"
ARQ_MODELO_INTENCAO = DIR_MODELOS / "classificador_intencao.joblib"

DIR_RESULTADOS = RAIZ / "resultados"
DIR_GRAFICOS = DIR_RESULTADOS / "graficos"
DIR_EXEMPLOS = DIR_RESULTADOS / "exemplos"
DIR_METRICAS = DIR_RESULTADOS / "metricas"
DIR_CACHE = DIR_PROCESSADO / "cache"

URL_TRASHNET = ("https://raw.githubusercontent.com/garythung/trashnet/"
                "master/data/dataset-resized.zip")
URL_ONNX = ("https://media.githubusercontent.com/media/onnx/models/main/"
            "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx")

SEMENTE = 42

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASSES_PT = {
    "cardboard": "papelão",
    "glass": "vidro",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "trash": "rejeito",
}

ARQ_LIMIARES = DIR_MODELOS / "limiares.json"
LIMIAR_CONFIANCA_ALTA = 0.70
LIMIAR_CONFIANCA_MINIMA = 0.45
if ARQ_LIMIARES.exists():
    import json as _json
    _lim = _json.loads(ARQ_LIMIARES.read_text(encoding="utf-8"))
    LIMIAR_CONFIANCA_ALTA = _lim["alta"]
    LIMIAR_CONFIANCA_MINIMA = _lim["minima"]
