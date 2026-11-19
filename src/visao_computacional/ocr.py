"""OCR de rótulos e embalagens com Tesseract (Smith, 2007)."""
import io
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

CONF_MIN = 60
IDIOMAS = "por+eng"


def disponivel() -> bool:
    return shutil.which("tesseract") is not None


def preprocessar(img_bgr: np.ndarray) -> np.ndarray:
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cinza = cv2.resize(cinza, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def ler_texto(img_bgr: np.ndarray, psm: int = 11, preprocessa: bool = True) -> dict:
    """Retorna {'texto': str, 'palavras': [(palavra, conf)]}."""
    if not disponivel():
        return {"texto": "", "palavras": [], "erro": "tesseract não instalado"}
    entrada = preprocessar(img_bgr) if preprocessa else img_bgr
    ok, png = cv2.imencode(".png", entrada)
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", IDIOMAS, "--psm", str(psm), "tsv"],
        input=png.tobytes(), capture_output=True, timeout=60)
    tsv = proc.stdout.decode("utf-8", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(tsv), sep="\t", quoting=3, dtype={"text": str})
    except Exception:
        return {"texto": "", "palavras": []}
    df = df[(df["conf"].astype(float) >= CONF_MIN) & df["text"].notna()]
    df = df[df["text"].str.strip().str.len() > 1]
    palavras = [(str(t).strip(), float(c)) for t, c in zip(df["text"], df["conf"])]
    return {"texto": " ".join(p for p, _ in palavras), "palavras": palavras}
