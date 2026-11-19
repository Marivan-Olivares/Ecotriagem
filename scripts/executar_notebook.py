"""Executa notebooks/analise_exploratoria.ipynb e salva as saídas no próprio arquivo."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
caminho = RAIZ / "notebooks" / "analise_exploratoria.ipynb"
nb = nbformat.read(caminho, as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(RAIZ / "notebooks")}}).execute()
nbformat.write(nb, caminho)
print(f"notebook executado: {caminho.relative_to(RAIZ)}")
sys.exit(0)
