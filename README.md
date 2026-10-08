# Skin cancer diagnosis using dermatological image analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-B.Eng._Thesis-success)

This repository contains the code and methodology for my Engineering Thesis: *"Skin cancer diagnosis using dermatological image analysis"*.

> ⚠️ **Language Note:**
> This project was developed at the Cracow University of Technology. While the code structure and logic are universal, **notebook comments and visualization labels are in Polish**. This README provides a complete English overview of the methodology and results to guide international readers.

## 📌 Project Overview
The goal of this research was to enhance the classification performance of skin lesion images (ISIC 2019 dataset) by combining a **DenseNet-121** backbone with classical Machine Learning classifiers (**SVM**, **Random Forest**).

Unlike standard approaches that rely solely on accuracy metrics, this study emphasizes **statistical rigor** (Friedman & Nemenyi tests) to validate model superiority.

## 📂 Project Structure & Translation Guide

The analysis is divided into 4 stages. Here is how to navigate the Polish notebooks:

### 1. Data Preparation (`01_data_preparation.ipynb`)
* **Goal:** Preprocessing ISIC 2019 dataset.
* **Key Technique:** Implemented **Stratified Group K-Fold**.
* *Why?* To strictly prevent data leakage by ensuring all images from the same patient (PatientID) remain in the same fold.

### 2. DenseNet Training (`02_densenet_training.ipynb`)
* **Goal:** Fine-tuning the **DenseNet-121** architecture.
* **Details:** Transfer learning implementation using PyTorch with custom training loops.

### 3. Hybrid Models (`03_hybrid_models_training.ipynb`)
* **Goal:** Feature extraction and Hybrid Classification.
* **Method:**
    1.  The classification layer (Softmax) of DenseNet is removed.
    2.  Deep features are extracted from images.
    3.  **SVM** and **Random Forest** classifiers are trained on these features.

### 4. Statistical Comparison (`04_statistical_comparison.ipynb`)
* **Goal:** Rigorous validation.
* **Key Visualizations (in Polish):**
    * *Macierz pomyłek* = **Confusion Matrix**
    * *Krzywa ROC* = **ROC Curve**
* **Method:** Used **Friedman Test** (to detect differences) and **Nemenyi Post-hoc Test** (to identify superior models).

## 📉 Key Results
The research led to the following conclusions:
1.  **Hybrid Models (DenseNet + SVM)** demonstrated statistically significant superiority over the standard Softmax approach.
2.  The hybrid approach proved to be more **robust** in borderline cases (where visual features were ambiguous).
3.  Statistical testing confirmed that the improvement was not due to random chance.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Machine Learning:** Scikit-learn (SVM, Random Forest)
* **Statistics:** Scikit-posthocs (Nemenyi test, Friedman test)
* **Data:** Pandas, NumPy, StratifiedGroupKFold
* **Visualization:** Matplotlib, Seaborn

---
*Author: Rafał Wodziński*

