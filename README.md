# High-Accuracy-CNN-Deep-Learning-for-Botanical-Image-Classification-Segmentation
Deep learning project using a scratch-built CNN for high-accuracy classification (95.8%) and a fine-tuned DeepLabv3+ model for precise segmentation (98.8%). Optimise the performance for small and complex botanical image datasets, using custom CNN architecture designs

## Deep Vision: Building CNNs from Scratch & Fine-Tuning for Flower Classification & Segmentation

This project applies **Deep Learning for computer vision** to solve two real-world image analysis tasks:  
1. **Multiclass Flower Classification** – identifying 17 species of flowers.  
2. **Semantic Segmentation** – detecting and isolating flower regions from the background.  

It combines a **scratch-built CNN** for classification and a **fine-tuned pretrained DeepLabv3+** model for segmentation.  
The work demonstrates how **custom CNN architectures** and **transfer learning** can both be effective for solving vision problems on small datasets.


---
# High-Accuracy-CNN-Deep-Learning-for-Botanical-Image-Classification-Segmentation

Deep learning project applying **Convolutional Neural Networks (CNNs)** to tackle two key computer vision tasks in botanical image analysis:  
1. **Multiclass Flower Classification** – built a **custom CNN from scratch** using MATLAB’s Deep Learning Toolbox, achieving **95.8% test accuracy** on the Oxford 17 Flowers dataset.  
2. **Semantic Segmentation** – fine-tuned a **pretrained DeepLabv3+ (ResNet-18 backbone)** for binary flower-vs-background segmentation, reaching **98.8% pixel accuracy** and **84.7% mean IoU** on the DaffodilSeg subset.

Designed to perform **optimally on small, complex datasets**, this project combines **custom CNN architecture design**, **data augmentation**, and **transfer learning** to deliver robust, high-accuracy models for real-world image classification and segmentation challenges.


---

## 🔍 Methodology Overview

- **Data Preparation**
  - Oxford 17 Flowers dataset reorganized into class folders.
  - DaffodilSeg subset prepared for segmentation; non-flower labels remapped to background.
  - 80/20 training-validation split with balanced classes.

- **Data Augmentation**
  - Rotations (±40°), translations (±20%), scaling (80–120%), shearing, and horizontal flips.
  - Standardized input size to **256×256×3**.

- **Classification Model**
  - Custom CNN with **4 Conv-ReLU-MaxPool blocks** (32–256 filters), dropout (0.3 & 0.5), fully connected layers, and softmax output.
  - Trained for **80 epochs** with Adam optimizer (`lr=1e-4`, batch size=32).

- **Segmentation Model**
  - Fine-tuned **DeepLabv3+** with **ResNet-18** backbone.
  - Binary segmentation (flower vs background).
  - Trained for **50 epochs** with Adam optimizer (`lr=1e-4`, batch size=8).

- **Evaluation Metrics**
  - Accuracy, precision, recall, F1-score, Intersection-over-Union (IoU), boundary F-score.
  - Confusion matrices for classification and segmentation.

---

## 📊 Results & Proof

### **Summary**

| Task            | Model                  | Accuracy / IoU  | Key Features                  |
|-----------------|------------------------|-----------------|--------------------------------|
| Classification  | Scratch-built CNN      | 95.81% accuracy | Data augmentation, dropout    |
| Segmentation    | DeepLabv3+ (ResNet-18) | 98.84% accuracy / 84.7% mIoU | Multi-scale context extraction |

### **1. Flower Classification (Scratch-built CNN)**
- **Validation accuracy (during training):** 79.78%  
- **Test accuracy (full dataset):** **95.81%**
- **Precision / Recall / F1-score:** 0.96 / 0.96 / 0.96
- **Key Insight:** Model generalizes well despite small dataset, thanks to extensive augmentation.

**Confusion Matrix Observation:**
- Misclassifications mostly between visually similar species (e.g., Daffodil vs Dandelion).
- Unique species with distinct color/shape reached near-perfect classification.

---

### **2. Flower Segmentation (Fine-tuned DeepLabv3+)**
- **Pixel Accuracy:** **98.84%**  
- **Mean IoU:** **84.7%**  
- **Class-level IoU:**  
  - Flower: 95.57%  
  - Background: 98.45%
- **Mean Boundary F-score:** 95.9%
- **Key Insight:** Model preserved petal boundaries well, even with cluttered backgrounds.

**Visual Proof:**  
Overlay predictions closely matched ground truth masks, with minimal under-segmentation at petal edges.

---

## 💡 Impact & Problem-Solving Value

- **Automation of Botanical Image Analysis:**  
  Enables accurate identification and segmentation of flowers, reducing the need for manual labeling.
  
- **Adaptability to Other Domains:**  
  Methods can be directly applied to tasks like medical imaging (tumor segmentation), agriculture (crop health monitoring), and industrial defect detection.

- **Small Dataset Efficiency:**  
  Achieved high accuracy using under 1,500 images, showing strong results without requiring massive datasets or expensive compute.

- **Transferable Skills Demonstrated:**
  - CNN architecture design from scratch.
  - Transfer learning and fine-tuning.
  - Data preprocessing, augmentation, and evaluation.
  - Working with imbalanced datasets.

---

## 🛠 Tech Stack

- **Language:** MATLAB
- **Libraries/Toolboxes:** Deep Learning Toolbox, Image Processing Toolbox
- **Datasets:** Oxford 17 Flowers, DaffodilSeg subset

---

## 📷 Sample Outputs

### Classification Confusion Matrix
*(Insert image here)*

### Segmentation Results
*(Insert overlay image here)*

---

## 📄 License
MIT License – free to use, modify, and share.

**Author:** Ngoc Hoa Nguyen  
University of Nottingham, UK
