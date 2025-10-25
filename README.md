# 🖼️ Multimodal Scene Classification

> 🧠 Multimodal scene classification combining **visual** (CNNs: VGG19, ResNet50, InceptionV3) and **audio** (MFCC → LSTM) features.
> The project investigates **early, late and hybrid fusion strategies** and uses ensemble methods (bagging, voting) to boost performance — reaching **up to ~99% accuracy** with a hybrid/ensemble pipeline on the chosen dataset.

---

## 📘 Table of Contents

- Project overview  
- Objectives  
- Dataset  
- Methodology  
- Fusion strategies
- Experimental Results
- Evaluation Metrics  
- Tech Stack 
- Key Skills & Concepts  
- Results Summary  
- Conclusion  
- Future work
- Useful Links 
- Author

---

## 📘 Project overview

This project focuses on **multimodal scene classification**, combining **visual** and **audio** data to recognize environmental contexts.
The system integrates **image features** (from CNNs like VGG19, ResNet50, and InceptionV3) and **audio features** (MFCC processed with LSTM) using **fusion-based architectures** — early, late, and hybrid.

Through these approaches, we achieved **up to 99% accuracy** on the **Kaggle Scene Classification Dataset**, demonstrating the effectiveness of combining complementary modalities.

---

## 🎯 Objectives

* **Visual Feature Extraction:** Use pre-trained CNNs (VGG19, ResNet50, InceptionV3) on ImageNet.
* **Audio Feature Extraction:** Derive MFCCs and learn temporal dependencies using LSTM networks.
* **Multimodal Fusion:** Explore early, late, and hybrid fusion architectures.
* **Performance Evaluation:** Compare models via Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
* **Ensemble Learning:** Implement bagging and majority voting to improve robustness.

---

## 🧩 Dataset

This project uses the **Kaggle “Scene Classification: Images and Audio”** dataset, which includes **images** paired with **MFCC audio features**.

📌 Dataset link:
[https://www.kaggle.com/code/kerneler/starter-scene-classification-images-40e223fe-3/](https://www.kaggle.com/code/kerneler/starter-scene-classification-images-40e223fe-3/)

### 📊 Dataset Summary

* **17,252 total samples**
* Each sample contains:

  * ✅ One environment **image**
  * ✅ Associated **audio MFCC features**
* **9 scene classes**, such as:
  *Beach, City, Classroom, Restaurant, Forest, …*

### 📁 Data Structure

```
dataset/
│
├── images/                  # All scene images grouped by class
│   ├── beach/
│   ├── forest/
│   ├── restaurant/
│   └── ...
│
└── dataset.csv              # image paths + MFCC features + class labels
      ├── image paths
      ├── mfcc_1 ... mfcc_104
      ├── CLASS1 (main label)
      └── CLASS2 (detailed label)
```

🔎 Each row in the CSV links an (image → MFCC audio → scene category), enabling **multimodal learning**.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

* **Images:**

  * Resized and normalized.
  * Denoising (median filter) and contrast enhancement.
* **Audio:**

  * Loaded from CSV, normalized, label-encoded, and aligned with image data.

### 2️⃣ Feature Extraction

* **Visual:** CNN-based embeddings from VGG19, ResNet50, InceptionV3 (fc2 layer).
* **Audio:** Temporal features using MFCC + LSTM layers.

### 3️⃣ Model Architectures

#### 🌀 Late Fusion

* Independent CNN and LSTM training.
* Fusion by:

  * **Average** of probabilities.
  * **Bagging** (Random Forest) on predicted outputs.

#### ⚡ Early Fusion

* Concatenation of image and audio features.
* Classifiers: **Dense NN**, **SVM (RBF kernel)**, and **Random Forest**.

#### 🚀 Hybrid Fusion

* Intermediate fusion of feature representations.
* Combines benefits of early (joint feature learning) and late (robust decision) fusions.

---

## 🔁 Fusion strategies

- Early (feature-level) fusion: concatenate embeddings before classification. Good for models that can learn joint representations.
  - Concatenate visual + audio embeddings and train classifiers (NN / SVM / RF). 
- Late (decision-level) fusion: combine model outputs (probabilities) — simple averaging or stacked/ensemble classifier. Useful when modalities have independent strengths.
  - train image and audio models independently and combine their predictions (average, voting, or bagging features into a meta-classifier). 
- Hybrid fusion: mix early and late fusion to capture cross-modal interactions and model-specific strengths; often yields the best trade-off.
  - split training set to train unimodal models and a combined model; average or stack predictions for final decision.

---

## 🧠 Experimental Results

| **Fusion Type** | **Best Model**   | **Feature Extractor**   | **Accuracy** | **Notes**                                  |
| --------------- | ---------------- | ----------------------- | ------------ | ------------------------------------------ |
| Late Fusion     | Average Ensemble | VGG19 + LSTM            | **0.98**     | Consistent across all classes              |
| Late Fusion     | Bagging (RF)     | VGG19 + LSTM            | 0.97         | Slightly lower, strong recall              |
| Early Fusion    | Dense NN         | VGG19 + MFCC-LSTM       | 0.96         | Great balance between precision and recall |
| Early Fusion    | SVM              | VGG19 + MFCC-LSTM       | 0.96         | Best for “Restaurant” class                |
| Early Fusion    | Random Forest    | VGG19 + MFCC-LSTM       | 0.95         | Stable across environments                 |
| Hybrid Fusion   | Dense NN         | InceptionV3 + MFCC-LSTM | **0.99**     | Top performer — best multimodal synergy    |

✅ **Highest accuracy (99%)** obtained with **InceptionV3 + LSTM (Hybrid Fusion)**.
**Confusion reduction** achieved by combining complementary modalities (visual + temporal).

---

## 📊 Evaluation Metrics

* **Accuracy (overall correctness)**
* **Precision & Recall (per class performance)**
* **F1-Score (harmonic mean of precision/recall)**
* **Confusion Matrix (error analysis)**

All metrics confirm the **superiority of hybrid fusion**, particularly for visually similar categories (e.g., *Forest vs Jungle*).

---

## 🧰 Tech Stack

* **Languages & Libraries:**

  * 🐍 Python, NumPy, Pandas, Matplotlib
  * 🎛️ TensorFlow, Keras, Scikit-learn
  * 🧩 OpenCV, Librosa (for MFCC)
* **Models:**

  * CNN (VGG19, ResNet50, InceptionV3)
  * LSTM (Audio temporal features)
  * Dense Neural Networks, SVM, Random Forest
* **Tools:** Jupyter Notebook, Kaggle, Google Colab

---

## 🚀 Key Skills & Concepts

* Transfer Learning (VGG19, InceptionV3)
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN, LSTM)
* MFCC Feature Extraction
* Early, Late & Hybrid Multimodal Fusion
* Ensemble Learning (Bagging, Voting)
* Model Evaluation & Visualization

---

## 📈 Results Summary

> “Multimodal learning leads to richer feature representation — achieving **human-like perception** through complementary cues.”

| Model                       | Accuracy | Precision | Recall   | F1-score |
| --------------------------- | -------- | --------- | -------- | -------- |
| CNN (VGG19)                 | 0.95     | 0.94      | 0.95     | 0.95     |
| LSTM (Audio)                | 0.94     | 0.94      | 0.94     | 0.94     |
| Hybrid (InceptionV3 + LSTM) | **0.99** | **0.99**  | **0.99** | **0.99** |

---

## 🏁 Conclusion

This project demonstrates that **combining visual and auditory modalities** significantly improves scene classification accuracy compared to unimodal approaches.
The **hybrid fusion strategy** effectively leverages both spatial and temporal cues, proving the potential of multimodal deep learning for complex perception tasks.

---

## 🎬 Demo

> ▶️ See the system classify a scene using **both image and audio cues**  
> 🌍 **Multimodal Scene Classification – Demo Video**  
> 🔗 [https://github.com/abdessamad-chahbi/multimodal-scene-classification/blob/main/demo.mp4](https://github.com/abdessamad-chahbi/multimodal-scene-classification/blob/main/demo.mp4)  

---

## 🔮 Future work

- Explore attention-based multimodal fusion (cross-modal transformers).  
- Fine-tune visual backbones end-to-end with multi-task losses.  
- Add temporal modeling across frames (3D CNNs / temporal attention) for video sequences.  
- Deploy a small inference API for real-time multimodal classification.

---

## 🔗 Useful Links

* 🧾 Dataset: [Scene Classification (Images & Audio) – Kaggle](https://www.kaggle.com/code/kerneler/starter-scene-classification-images-40e223fe-3/)
* 💻 Project Repository: [GitHub – abdessamad-chahbi/multimodal-scene-classification](https://github.com/abdessamad-chahbi/multimodal-scene-classification)

---

## 👤 Author

**Abdessamad CHAHBI**
• 🌐 [Portfolio](https://abdessamad-chahbi.github.io)
• [LinkedIn](https://linkedin.com/in/abdessamad-chahbi) 
• [GitHub](https://github.com/abdessamad-chahbi)

---
