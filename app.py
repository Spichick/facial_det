import cv2
import joblib
import gradio as gr
import numpy as np
from skimage.feature import local_binary_pattern

# how to run:
# 1. python app.py
# 2. open http://
# 3. upload an image to see the result
knn = joblib.load("models/lbp_knn.pkl")

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def extract_lbp_feature(gray, P=10, R=2, method="uniform"):
    lbp = local_binary_pattern(gray, P, R, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist

def predict_expression(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (48,48)) 

    feat = extract_lbp_feature(face).reshape(1, -1)
    y_pred = knn.predict(feat)[0]

    if hasattr(knn, "predict_proba"):
        proba = knn.predict_proba(feat)[0]
        conf = float(proba.max())
    else:
        conf = 1.0

    label = CLASSES[int(y_pred)] if isinstance(y_pred, (int, np.integer)) else str(y_pred)
    return f"{label} | confidence={conf:.2f}"

# ========== interface ==========
demo = gr.Interface(
    fn=predict_expression,
    inputs=gr.Image(type="numpy", sources=["upload", "webcam"], label="Upload or Webcam"),
    outputs=gr.Textbox(label="LBP+KNN Result"),
    title="Facial Expression Recognition (LBP+KNN)"
)

if __name__ == "__main__":
    demo.launch()
