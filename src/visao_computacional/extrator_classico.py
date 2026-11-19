"""Características visuais clássicas (linha de base sem aprendizado profundo)."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def caracteristicas_classicas(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (256, 192), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist /= hist.sum() + 1e-9

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_feats = []
    for p, r in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(cinza, p, r, method="uniform")
        h, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
        lbp_feats.append(h / (h.sum() + 1e-9))

    hog_feat = hog(cv2.resize(cinza, (128, 96)), orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)

    bordas = cv2.Canny(cinza, 80, 160)
    estat = np.concatenate([hsv.reshape(-1, 3).mean(0) / 255, hsv.reshape(-1, 3).std(0) / 255,
                            [bordas.mean() / 255]])
    return np.concatenate([hist, *lbp_feats, hog_feat, estat]).astype(np.float32)


def extrair_lote(imagens_bgr) -> np.ndarray:
    return np.stack([caracteristicas_classicas(im) for im in imagens_bgr])
