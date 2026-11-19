import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import config


def _pronto():
    return config.ARQ_MODELO_VISAO.exists() and config.ARQ_ONNX.exists() \
        and config.ARQ_DIVISAO.exists()


precisa_modelos = pytest.mark.skipif(
    not _pronto(), reason="rode scripts/baixar_dados.py e o pipeline de treino antes")
