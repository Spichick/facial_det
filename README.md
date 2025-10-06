# HOW TO RUN

1. In terminal, run:
   ```bash
   python app.py

2. Open the link http:// using Left Ctrl + Left Click.

3. Upload an image, then click Run to see the result.


# Structure
```
facial_det/
│
├── models/
│   ├── cnn.h5
│   ├── hog_svm.pkl
│   ├── lbp_knn.pkl
│   │        → 3 trained models
│   │
│   ├── cnn.py
│   ├── hog_svm.py
│   └── lbp_knn.ipynb
│            → 3 Python codes that generate the models
│
├── utils/
│   ├── lbp_features.py
│   ├── lbp_knn.py
│
├── app.py                 → main program
├── emorec_pretrain.py
├── train_mini_xception.py
└── README.md
```
3 trained models, 3 Python codes that generate the models and the main program are useful, the other files are not required (Some are only for debug or give us insights).