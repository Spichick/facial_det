import numpy as np
import cv2
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

X = []
y = []

data_dir = "models/train"
for label in range(7):
    folder = os.path.join(data_dir, str(label))
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        
        feat = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
        X.append(feat)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = LinearSVC(C=0.01, max_iter=5000)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(svm, "models/hog_svm.pkl")
print("Model saved: models/hog_svm.pkl")
