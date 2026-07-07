
# CodeAlpha_HandwrittenCharacterRecognition

## 📌 Task
CodeAlpha Machine Learning Internship — Task 3: Handwritten Character Recognition

## 🎯 Objective
Recognize handwritten digits using a deep learning model trained on the MNIST dataset. The model takes an image of a handwritten digit (0–9) as input and predicts the correct digit.

## 🧠 Approach
- Loaded and preprocessed the MNIST dataset (normalization, reshaping for CNN input, one-hot encoding of labels)
- Built a **Convolutional Neural Network (CNN)** using TensorFlow/Keras with multiple Conv2D and MaxPooling layers
- Trained the model over 10 epochs with a validation split
- Evaluated performance using accuracy, loss curves, and a confusion matrix
- Saved the trained model for future use/inference

## 📂 Project Structure
```
CodeAlpha_HandwrittenCharacterRecognition/
│
├── handwritten_character_recognition.py   # Main script: data loading, model building, training, evaluation
├── handwritten_character_recognition_model.h5   # Saved trained CNN model
├── sample_digits.png                      # Sample images from the MNIST dataset
├── training_history.png                   # Accuracy & loss curves over epochs
├── confusion_matrix.png                   # Confusion matrix on test data
└── README.md
```

## 📊 Results
- Model type: CNN (3 convolutional layers + fully connected layers with dropout)
- Achieved high test accuracy (~98%+) on unseen MNIST test data
- Confusion matrix shows strong per-class performance across all 10 digit classes

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## ▶️ How to Run
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
python handwritten_character_recognition.py
```

## 📈 Possible Extensions
- Extend to EMNIST for full alphabet character recognition
- Extend to full word/sentence recognition using sequence models like CRNN


