"""Extração de características profundas com MobileNetV2 (transfer learning)."""
import cv2
import numpy as np

from src import config

MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CANDIDATOS_SAIDA = ("472", "Reshape_103", "464")
TAMANHO = 224


class ExtratorMobileNet:
    dimensao = 1280

    def __init__(self, caminho_onnx=config.ARQ_ONNX):
        if not caminho_onnx.exists():
            raise FileNotFoundError(
                f"{caminho_onnx} não encontrado. Rode: python scripts/baixar_dados.py")
        self.rede = cv2.dnn.readNetFromONNX(str(caminho_onnx))
        self.camada = None
        teste = self._blob([np.zeros((TAMANHO, TAMANHO, 3), np.uint8)])
        for nome in CANDIDATOS_SAIDA:
            try:
                self.rede.setInput(teste)
                if self.rede.forward(nome).size == self.dimensao:
                    self.camada = nome
                    break
            except cv2.error:
                continue
        if self.camada is None:
            raise RuntimeError("Camada de embedding da MobileNetV2 não encontrada.")

    @staticmethod
    def _blob(imagens_bgr):
        lote = []
        for img in imagens_bgr:
            rgb = cv2.cvtColor(cv2.resize(img, (TAMANHO, TAMANHO),
                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - MEDIA) / DESVIO
            lote.append(x.transpose(2, 0, 1))
        return np.stack(lote)

    def extrair(self, imagens_bgr, lote: int = 32) -> np.ndarray:
        if isinstance(imagens_bgr, np.ndarray) and imagens_bgr.ndim == 3:
            imagens_bgr = [imagens_bgr]
        saidas = []
        for i in range(0, len(imagens_bgr), lote):
            self.rede.setInput(self._blob(imagens_bgr[i:i + lote]))
            saidas.append(self.rede.forward(self.camada).reshape(-1, self.dimensao))
        return np.vstack(saidas).astype(np.float32)
