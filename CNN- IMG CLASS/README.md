# Pediatric Pneumonia Classification using CNNs

## 📌 Project Overview
This project aims to develop a Convolutional Neural Network (CNN) to automatically classify chest X-ray images into two categories: **Pneumonia** and **Normal**. The dataset comprises 5,863 JPEG anterior-posterior chest X-ray images sourced from pediatric patients (aged one to five years) as part of their routine clinical care at a renowned hospital.

As a healthcare data science initiative, the primary goal is to accurately identify the presence of pneumonia from raw medical imaging, assisting clinicians in rapid and reliable diagnostics.

## 💾 Dataset Setup & Google Drive Integration

**IMPORTANT:** The code is designed to run in Google Colab and expects the dataset to be hosted in your Google Drive. 

To set up the project:
1. Upload the dataset folder to your Google Drive. 
2. Ensure the folder structure is exactly as follows:
   ```text
   My Drive/
   └── Archive/
       ├── train/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       ├── val/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── test/
           ├── NORMAL/
           └── PNEUMONIA/
   ```
3. When running the notebook, it will prompt you to mount your Google Drive using `drive.mount('/content/drive')` to securely access these files.

## 🔬 Approach
Medical imaging datasets often present unique challenges, such as class imbalance and high intra-class variance. Our approach utilizes **Deep Learning**, specifically a custom-built Convolutional Neural Network (CNN), which excels at spatial feature extraction (e.g., detecting lung opacities or rib structures) directly from pixel arrays. 

Instead of manually extracting features, the CNN learns hierarchical representations of the X-rays through backpropagation. To address the relatively limited size of medical datasets, we utilize **Data Augmentation** to synthetically expand the training data and improve the model's generalization capabilities.

## 🛠 Methodology

### 1. Data Preprocessing & Augmentation
* **Rescaling:** All image pixel values (0-255) are normalized to a scale of 0 to 1 to ensure stable and faster convergence during training.
* **Data Augmentation:** Applied exclusively to the `train` dataset. We applied random zooming (20%), horizontal flipping, and rotation (10 degrees). This prevents the model from memorizing exact pixel locations and forces it to learn underlying anatomical patterns.
* **Integrity:** The `val` and `test` sets are intentionally *not* augmented to provide an unbiased evaluation of the model on pristine, real-world data.

### 2. Model Architecture
The custom CNN is built using the TensorFlow/Keras `Sequential` API:
* **Convolutional Blocks:** Four successive blocks of `Conv2D` layers (increasing filters: 32 → 64 → 128 → 128) paired with `MaxPooling2D` layers. This progressively extracts higher-level features while reducing spatial dimensions.
* **Classification Head:** A `Flatten` layer followed by a dense network of 512 neurons. 
* **Regularization:** A `Dropout` layer (rate = 0.5) is implemented to randomly deactivate neurons during training, drastically reducing overfitting.
* **Output:** A single-neuron `Dense` layer with a `sigmoid` activation function, outputting a probability between 0 and 1 (0 = Normal, 1 = Pneumonia).

### 3. Training Strategy
* **Optimizer & Loss:** The model is optimized using the `adam` optimizer and evaluates error using `binary_crossentropy`.
* **Callbacks:** 
  * `EarlyStopping`: Halts training if validation loss stops improving for 5 consecutive epochs, restoring the best model weights.
  * `ModelCheckpoint`: Saves the ultimate best version of the model to disk (`best_pneumonia_model.h5`).

### 4. Evaluation Metrics
In a clinical setting, **False Negatives** (diagnosing a sick patient as healthy) are highly dangerous. Therefore, beyond raw accuracy, the model is evaluated using:
* **Confusion Matrix:** To visually track False Positives and False Negatives.
* **Classification Report:** To measure Precision, Recall (Sensitivity), and F1-score for both classes individually. The test data is evaluated with `batch_size=1` and `shuffle=False` to ensure predictions align perfectly with true labels.

## 📊 Expected Findings & Clinical Viability
* **High Sensitivity:** CNNs trained on this dataset typically achieve high Recall for the Pneumonia class, meaning the model is highly sensitive to detecting the disease and minimizes dangerous False Negatives.
* **Feature Generalization:** Thanks to the dropout layers and image augmentation, the model successfully generalizes beyond the training data, exhibiting stable accuracy and loss curves during validation.
* **Inference Capability:** The final implementation includes an inference function (`predict_xray`) that allows a practitioner to input a single new X-ray image and receive an immediate diagnostic prediction along with a confidence score.

---
*Built using TensorFlow, Keras, NumPy, and Matplotlib.*
