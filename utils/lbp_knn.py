# models/lbp_knn.py
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from .lbp_features import lbp_hist

class LbpKnnModel:
    """
    统一接口：
      - fit(X_faces_gray, y)
      - predict_one(face_gray48) -> (label, confidence)
      - predict_batch(faces_gray48) -> (labels, confidences)
      - save(path), load(path)
    """
    def __init__(self, n_neighbors=5, weights="distance", metric="euclidean"):
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,  
            metric=metric
        )
        self.is_fitted = False

    def _extract_one(self, face_gray):
        return lbp_hist(face_gray)

    def _extract_batch(self, faces_gray):
        return np.stack([self._extract_one(f) for f in faces_gray], axis=0)

    def fit(self, X_faces_gray, y):
        """
        X_faces_gray: List/ndarray of gray faces (H=W=48)
        y: labels (list/ndarray),可为int或str
        """
        X = self._extract_batch(X_faces_gray)
        self.knn.fit(X, y)
        self.is_fitted = True
        return self

    def predict_one(self, face_gray48):
        """
        返回: (label, confidence[0..1])
        """
        x = self._extract_one(face_gray48)[None, :]
        yhat = self.knn.predict(x)[0]

        conf = None
        if hasattr(self.knn, "predict_proba"):
            proba = self.knn.predict_proba(x)[0]
            conf = float(proba.max())
        #if conf is None:
        #    dists, idx = self.knn.kneighbors(x, n_neighbors=self.knn.n_neighbors, return_distance=True)
        #    d = dists[0]
        #    w = 1.0 / (d + 1e-6)
        #    conf = float((w.max() / (w.sum() + 1e-8)))

        return yhat, conf

    def predict_batch(self, faces_gray48):
        X = self._extract_batch(faces_gray48)
        yhat = self.knn.predict(X)
        confs = None
        if hasattr(self.knn, "predict_proba"):
            proba = self.knn.predict_proba(X)
            confs = proba.max(axis=1).astype(float)
        else:
            confs = np.ones(len(yhat), dtype=float)
        return yhat, confs

    def save(self, path):
        joblib.dump(self.knn, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.knn = joblib.load(path)
        obj.is_fitted = True
        return obj


def load_model(path):
    global _model_singleton
    _model_singleton = LbpKnnModel.load(path)
    return _model_singleton

def predict(face_gray48):
    """
    给 UI 的统一入口：返回 (label, confidence)
    """
    assert _model_singleton is not None, "LBP+KNN model not loaded. Call load_model(path) first."
    return _model_singleton.predict_one(face_gray48)
