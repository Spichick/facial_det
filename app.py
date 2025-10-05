import cv2
import joblib
import gradio as gr
import numpy as np
import time
from skimage.feature import local_binary_pattern, hog
from tensorflow.keras.models import load_model

# how to run:
# 1. python app.py
# 2. open the local link shown (e.g. http://127.0.0.1:7860)
# 3. upload an image or use webcam to see the results

# ===========================
# Load Models
# ===========================
lbp_knn = joblib.load("models/lbp_knn.pkl")       # LBP + KNN
hog_svm = joblib.load("models/hog_svm.pkl")       # HOG + SVM
cnn_model = load_model("models/cnn.h5") # CNN (trained model)

print("✅ Models loaded successfully: LBP+KNN, HOG+SVM, CNN")

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ===========================
# Feature Extraction
# ===========================
def extract_lbp_feature(gray, P=10, R=2, method="uniform"):
    """Extract LBP histogram as feature vector."""
    lbp = local_binary_pattern(gray, P, R, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist


# ===========================
# Prediction Functions
# ===========================

def predict_lbp_knn(face_gray):
    """Predict using LBP+KNN."""
    start_time = time.time()
    try:
        face = cv2.resize(face_gray, (48, 48))
        feat = extract_lbp_feature(face).reshape(1, -1)
        y_pred = lbp_knn.predict(feat)[0]
        conf = float(np.max(lbp_knn.predict_proba(feat))) if hasattr(lbp_knn, "predict_proba") else 1.0
        label = CLASSES[int(y_pred)] if isinstance(y_pred, (int, np.integer)) else str(y_pred)
        latency = (time.time() - start_time) * 1000
        return label, conf, latency
    except Exception as e:
        return f"error: {str(e)}", 0.0, 0.0


def predict_hog_svm(face_gray):
    """Predict using HOG + Linear SVM."""
    start_time = time.time()
    try:
        face = cv2.resize(face_gray, (48, 48))
        feat = hog(face, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
        feat = feat.reshape(1, -1)
        y_pred = hog_svm.predict(feat)[0]
        label = CLASSES[int(y_pred)] if isinstance(y_pred, (int, np.integer)) else str(y_pred)
        latency = (time.time() - start_time) * 1000
        return label, 1.0, latency
    except Exception as e:
        return f"error: {str(e)}", 0.0, 0.0


def predict_cnn(face_gray):
    """Predict using trained CNN (.h5)."""
    start_time = time.time()
    try:
        face = cv2.resize(face_gray, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # shape (1,48,48,1)
        preds = cnn_model.predict(face, verbose=0)[0]
        label_idx = np.argmax(preds)
        conf = float(preds[label_idx])
        label = CLASSES[label_idx]
        latency = (time.time() - start_time) * 1000
        return label, conf, latency
    except Exception as e:
        return f"error: {str(e)}", 0.0, 0.0


# ===========================
# Main Wrapper (face detection + model calls)
# ===========================

def unified_prediction(img):
    """Run face detection and all three model predictions."""
    if img is None:
        return "No image", "No image", "No image"

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No faces detected", "No faces detected", "No faces detected"

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    face_gray = gray[y:y+h, x:x+w]

    # Predictions
    lbp_label, lbp_conf, lbp_lat = predict_lbp_knn(face_gray)
    hog_label, hog_conf, hog_lat = predict_hog_svm(face_gray)
    cnn_label, cnn_conf, cnn_lat = predict_cnn(face_gray)

    # Format outputs
    lbp_text = f"{lbp_label} | conf={lbp_conf:.2f} | {lbp_lat:.1f}ms"
    hog_text = f"{hog_label} | conf={hog_conf:.2f} | {hog_lat:.1f}ms"
    cnn_text = f"{cnn_label} | conf={cnn_conf:.2f} | {cnn_lat:.1f}ms"

    return lbp_text, hog_text, cnn_text


# ===========================
# Gradio Interface
# ===========================
with gr.Blocks() as demo:
    gr.Markdown("# Facial Expression Recognition (LBP+KNN vs HOG+SVM vs CNN)")
    gr.Markdown("Upload an image or use webcam to see predictions from all models")

    with gr.Row():
        input_image = gr.Image(type="numpy", sources=["upload", "webcam"], label="Upload or Webcam")

    with gr.Row():
        lbp_output = gr.Textbox(label="LBP+KNN Result", lines=2)
        hog_output = gr.Textbox(label="HOG+SVM Result", lines=2)
        cnn_output = gr.Textbox(label="CNN Result", lines=2)

    input_image.change(
        fn=unified_prediction,
        inputs=input_image,
        outputs=[lbp_output, hog_output, cnn_output]
    )

if __name__ == "__main__":
    demo.launch()
