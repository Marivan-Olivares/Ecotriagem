#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3 || command -v python)}"

echo "== 1/9 download (TrashNet + MobileNetV2 ONNX)";   "$PY" scripts/baixar_dados.py
echo "== 2/9 limpeza e divisão dos dados";            "$PY" scripts/preparar_dados.py
echo "== 3/9 dataset de intenções (PLN)";             "$PY" scripts/gerar_dataset_pln.py
echo "== 4/9 extração de características";            "$PY" scripts/extrair_caracteristicas.py
echo "== 5/9 treino/seleção dos modelos de visão";    "$PY" scripts/treinar_visao.py
echo "== 6/9 treino/avaliação do PLN";                "$PY" scripts/treinar_pln.py
echo "== 7/9 experimentos (E1-E7)";                   "$PY" scripts/experimentos.py
echo "== 8/9 demonstração do fluxo integrado";        "$PY" demo.py --roteiro
echo "== 9/9 testes automatizados";                   "$PY" -m pytest
if command -v dot >/dev/null; then
  dot -Tpng -Gdpi=200 docs/arquitetura.dot -o resultados/graficos/arquitetura.png
fi
"$PY" scripts/executar_notebook.py
echo "Concluído. Métricas em resultados/metricas, gráficos em resultados/graficos."
