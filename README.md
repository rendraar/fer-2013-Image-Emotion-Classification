# Facial Expression Recognition with CNN and Transfer Learning

## Deskripsi Project
Project ini bertujuan untuk melakukan **klasifikasi ekspresi wajah** menggunakan dataset **FER-2013**. Model yang digunakan terdiri dari:
1. **Base CNN** (from scratch) 
2. **VGG16 Pretrained** (Transfer Learning)  
3. **EfficientNetB0 Pretrained** (Transfer Learning)  

Setiap model dievaluasi menggunakan:
- **Accuracy**  
- **Precision, Recall, F1-score** (Classification Report)  
- **Confusion Matrix**  
- **Grafik Loss dan Accuracy**  

---

## Dataset
Dataset yang digunakan adalah **FER-2013 (Facial Expression Recognition 2013)**:

- Format: Gambar grayscale 48x48  
- Jumlah kelas: 7 (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`)  
- Link dataset publik: [FER-2013 Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  

---

## Base Model CNN
### Konfigurasi
Model CNN dasar dibangun **tanpa pretrained weights**, menggunakan gambar **grayscale 48×48** sesuai format asli FER-2013.

**Detail konfigurasi:**
- Input: **48×48 grayscale (1 channel)**
- Augmentasi data (training):
  - Rotation range: 10°
  - Width & height shift: 0.1
  - Horizontal flip
- Validation split: **20% dari data training**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 40
- Batch size: 64

**Arsitektur Model:**
- Conv2D (32) → BatchNormalization → MaxPooling
- Conv2D (64) → BatchNormalization → MaxPooling
- Conv2D (128) → BatchNormalization → MaxPooling
- Flatten
- Dense (128, ReLU)
- Dropout (0.5)
- Dense (Softmax – 7 kelas)

---

### Hasil Evaluasi Base CNN
- **Accuracy:** **0.602**
- **Weighted F1-score:** **0.590**
- **Macro F1-score:** **0.508**

**File hasil evaluasi:**
- Confusion Matrix: `results/confusion_matrix_base_cnn.png`
- Grafik Loss & Accuracy: `results/loss_accuracy_base_cnn.png`
- Model tersimpan: `results/base_cnn_model.h5`

---

## Pretrained Model VGG16
### Setup Training
**Konfigurasi utama:**
- Input image size: **96×96 RGB**  
- Batch size: **32**
- Epochs:
  - Base training: **25**
  - Fine-tuning: **25**
- Optimizer:
  - Base: Adam (LR = 1e-4)
  - Fine-tune: Adam (LR = 1e-5)
- Loss: Categorical Crossentropy
- Class imbalance handling: **Class Weights**

**Augmentasi data:**
- Rotation: ±10°
- Width & height shift: 5%
- Horizontal flip
- Validation split: **20% dari data training**

---

### Arsitektur Model
- Base model: **VGG16 (include_top=False, pretrained ImageNet)**
- Semua layer VGG16 **dibekukan (freeze)** pada tahap awal
- Head classifier:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense (256, ReLU)
  - BatchNormalization
  - Dropout (0.5)
  - Dense (Softmax – 7 kelas)

**Fine-tuning:**
- 6 layer terakhir VGG16 dibuka (trainable)
- Learning rate diturunkan untuk stabilitas training

---

### Hasil Evaluasi VGG16
- **Accuracy:** **0.649**
- **Weighted F1-score:** **0.649**
- **Macro F1-score:** **0.626**

---

### File Hasil Evaluasi
- Classification Report: `results_vgg/vgg_classification_report.csv`
- Confusion Matrix: `results_vgg/vgg_confusion_matrix.png`
- Grafik Loss & Accuracy: `results_vgg/vgg_loss_accuracy.png`
- Model tersimpan: `results_vgg/vgg_model.h5`

---

## Pretrained Model EfficientNetB0
### Setup Training
**Konfigurasi utama:**
- Input image size: **128×128 RGB**
- Batch size: **16**
- Epochs: **40**
- Optimizer: Adam (Learning Rate = **1e-5**)
- Loss: Categorical Crossentropy
- Validation split: **20% dari data training**

**Augmentasi data:**
- Rotation: ±10°
- Width & height shift: 10%
- Horizontal flip
- Preprocessing: `efficientnet.preprocess_input`

---

### Arsitektur Model
- Base model: **EfficientNetB0 (include_top=False, pretrained ImageNet)**
- Seluruh layer EfficientNet **dibekukan (freeze)** selama training
- Head classifier:
  - GlobalAveragePooling2D
  - Dense (256, ReLU)
  - Dropout (0.5)
  - Dense (Softmax – 7 kelas)

---

### Hasil Evaluasi EfficientNetB0
- **Accuracy:** **0.447**
- **Weighted F1-score:** **0.411**
- **Macro F1-score:** **0.343**

---

### File Hasil Evaluasi
- Classification Report: `results_effnet/effnet_classification_report.csv`
- Confusion Matrix: `results_effnet/effnet_confusion_matrix.png`
- Grafik Loss & Accuracy: `results_effnet/effnet_loss_accuracy.png`
- Model tersimpan: `results_effnet/efficientnet_model.h5`  

---

## Tabel Perbandingan Model

| Nama Model     | Akurasi | Hasil Analisis |
|---------------|---------|----------------|
| Base CNN | **0.60** | Model CNN sederhana dengan input grayscale 48×48. Mampu mengenali kelas mayoritas seperti `happy` dan `surprise`, namun performa pada kelas minoritas terutama `disgust` dan `fear` masih sangat rendah. Augmentasi membantu stabilitas training, tetapi keterbatasan arsitektur membuat generalisasi kurang optimal. |
| VGG16 | **0.65** | Transfer learning memberikan peningkatan signifikan dibanding Base CNN. Model mampu menangkap fitur wajah yang lebih kompleks, dengan peningkatan recall dan F1-score pada hampir semua kelas, termasuk kelas minoritas (`disgust`, `fear`). Performa paling stabil dan seimbang di antara ketiga model. |
| EfficientNetB0 | **0.45** | Meskipun menggunakan arsitektur pretrained modern, performa relatif rendah. Model mengalami underfitting karena base model sepenuhnya dibekukan dan learning rate kecil. Beberapa kelas minoritas seperti `disgust` tidak terdeteksi (recall = 0), sehingga akurasi dan F1-score turun signifikan. |

---


## Cara Menjalankan Project
1. Clone repository:  
```bash
git clone <URL_REPO>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan Streamlit Dashboard:
```bash
streamlit run base_cnn_app.py
```

Gunakan tombol pada homepage untuk mengakses masing-masing model (Base CNN, VGG16, EfficientNet).


## Catatan
- GPU sangat disarankan untuk proses training model pretrained (VGG16 dan EfficientNetB0).
- Augmentasi data berperan penting dalam meningkatkan performa pada class minoritas seperti `disgust` dan `fear`.
- Jika GPU tidak tersedia, inference tetap dapat dijalankan menggunakan CPU.