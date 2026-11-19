"""Baixa o dataset TrashNet e o modelo MobileNetV2 (ONNX)."""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


def baixar(url: str, destino: Path) -> None:
    if destino.exists() and destino.stat().st_size > 0:
        print(f"[ok] já existe: {destino.relative_to(config.RAIZ)}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"[..] baixando {url}")
    tmp = destino.with_suffix(destino.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(destino)
    print(f"[ok] {destino.relative_to(config.RAIZ)} "
          f"({destino.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    zip_path = config.DIR_BRUTO / "dataset-resized.zip"
    baixar(config.URL_TRASHNET, zip_path)
    if not config.DIR_IMAGENS.exists():
        print("[..] extraindo imagens")
        with zipfile.ZipFile(zip_path) as z:
            membros = [m for m in z.namelist() if not m.startswith("__MACOSX")]
            z.extractall(config.DIR_BRUTO, members=membros)
    print(f"[ok] imagens em {config.DIR_IMAGENS.relative_to(config.RAIZ)}")
    baixar(config.URL_ONNX, config.ARQ_ONNX)


if __name__ == "__main__":
    main()
