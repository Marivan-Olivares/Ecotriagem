"""Modelos supervisionados candidatos para a classificação visual."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

from src import config


def svm_calibrado(metodo="sigmoid"):
    """SVM com kernel RBF + calibração para obter probabilidades."""
    return CalibratedClassifierCV(SVC(kernel="rbf", gamma="scale", class_weight="balanced"),
                                  method=metodo, cv=5, ensemble=(metodo == "isotonic"))


def candidatos():
    """Retorna {nome: (caracteristica, aumento, pipeline, grade_hiperparametros)}."""
    s = config.SEMENTE
    return {
        "Clássico + SVM-RBF": (
            "classica", False,
            Pipeline([("pad", StandardScaler()),
                      ("clf", svm_calibrado())]),
            {"clf__estimator__C": [1, 10]}),
        "Clássico + Random Forest": (
            "classica", False,
            Pipeline([("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                     n_jobs=-1, random_state=s))]),
            {"clf__max_features": ["sqrt", 0.05]}),
        "CNN + k-NN": (
            "cnn", False,
            Pipeline([("norm", Normalizer()), ("clf", KNeighborsClassifier(weights="distance"))]),
            {"clf__n_neighbors": [3, 7, 15]}),
        "CNN + SVM-RBF (Platt)": (
            "cnn", False,
            Pipeline([("pad", StandardScaler()), ("clf", svm_calibrado("sigmoid"))]),
            {"clf__estimator__C": [1, 10]}),
        "CNN + SVM-RBF (isotônica)": (
            "cnn", False,
            Pipeline([("pad", StandardScaler()), ("clf", svm_calibrado("isotonic"))]),
            {"clf__estimator__C": [1, 10]}),
        "CNN + Reg. Logística": (
            "cnn", False,
            Pipeline([("pad", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1.0]}),
        "CNN + Reg. Logística + aumento": (
            "cnn", True,
            Pipeline([("pad", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1.0]}),
    }
