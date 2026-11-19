"""Carregamento, padronização e perturbações de imagens (OpenCV)."""
import io

import cv2
import numpy as np


def carregar_imagem(origem) -> np.ndarray:
    """Aceita caminho, bytes, arquivo aberto, PIL.Image ou ndarray; devolve BGR uint8."""
    if isinstance(origem, np.ndarray):
        img = origem
    elif hasattr(origem, "convert"):
        img = cv2.cvtColor(np.array(origem.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        if hasattr(origem, "read"):
            origem = origem.read()
        if isinstance(origem, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(origem, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(origem), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível ler a imagem.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def padronizar(img: np.ndarray, largura: int = 512) -> np.ndarray:
    """Redimensiona mantendo a proporção (as fotos do TrashNet têm 512x384)."""
    h, w = img.shape[:2]
    if w == largura:
        return img
    escala = largura / w
    return cv2.resize(img, (largura, int(round(h * escala))), interpolation=cv2.INTER_AREA)


def desfoque(img, k=9):
    return cv2.GaussianBlur(img, (k, k), 0)


def ruido(img, sigma=20, semente=0):
    rng = np.random.default_rng(semente)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def brilho(img, fator=0.5):
    return np.clip(img.astype(np.float32) * fator, 0, 255).astype(np.uint8)


def rotacao(img, angulo=30):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def oclusao(img, fracao=0.25, semente=0):
    rng = np.random.default_rng(semente)
    h, w = img.shape[:2]
    lado_h, lado_w = int(h * np.sqrt(fracao)), int(w * np.sqrt(fracao))
    y, x = rng.integers(0, h - lado_h), rng.integers(0, w - lado_w)
    out = img.copy()
    out[y:y + lado_h, x:x + lado_w] = 127
    return out


def jpeg(img, qualidade=15):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


PERTURBACOES = {
    "original": lambda im: im,
    "desfoque": desfoque,
    "ruído gaussiano": ruido,
    "baixo brilho": brilho,
    "rotação 30°": rotacao,
    "oclusão 25%": oclusao,
    "JPEG q=15": jpeg,
}


def para_bytes_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes()).getvalue()
