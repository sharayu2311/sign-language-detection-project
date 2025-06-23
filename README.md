# 🤟 Real-Time Sign Language Detection

This project is a deep learning-based real-time sign language detection system using TensorFlow, Keras, OpenCV, and MobileNetV2. It classifies hand gestures captured from your webcam into six predefined categories.

---

## 📁 Dataset Structure

Ensure your dataset is organized as:

Sign_language_data/
├── train/
│   └── images/
│       ├── Hello/
│       ├── Iloveyou/
│       ├── No/
│       ├── Please/
│       ├── Thanks/
│       └── Yes/
├── test/
│   └── images/
│       └── (same folders as above)
└── data.yaml

Example data.yaml:
train: /absolute/path/to/train/images
val: /absolute/path/to/test/images
nc: 6
names: ['Hello', 'Iloveyou', 'No', 'Please', 'Thanks', 'Yes']

---

## 🧠 Model Overview

- Base Model: MobileNetV2 (pretrained on ImageNet)
- Architecture: MobileNetV2 + GlobalAvgPooling + Dense + Dropout + Softmax
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## 🎥 Real-Time Prediction

After training, the webcam is used to detect hand gestures. The predicted class with its probability is displayed live on the video frame.

> Press `q` to exit the detection window.

---

## 🚀 Usage

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the script:
   python final.py

---

## 📦 Dataset Storage Note

Due to GitHub's storage space limits, the dataset is divided into smaller parts:

- **Training data** is split into multiple folders:
  - `train_dataset_1`, `train_dataset_2`, etc.
- **Testing data** is split into:
  - `test_dataset_1`, `test_dataset_2`
- The corresponding label files (`train.labels`, `test.labels`) are also stored separately in different folders for better organization.

Make sure to merge or load them appropriately when training the model.

---

## 📦 Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- PyYAML
- NumPy
- Matplotlib

(See requirements.txt for version details)

---

## 👩‍💻 Author

Sharayu Bodkhe  
B.Tech CSE | Graphic Era University  
GitHub: @sharayubodkhe

---

## 📜 License

This project is licensed under the MIT License.
