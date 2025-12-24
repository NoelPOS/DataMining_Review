# Data Mining Review & Python Implementation

This repository contains Python implementations of Data Mining algorithms found in "Learning Data Mining with Python". The code has been refactored, organized by chapter, and updated to ensure portability and ease of use.

## 📂 Project Structure

```
DataMining_Review/
├── Chapter_01_Introduction/   # Hello World & Sklearn Basics
├── Chapter_02_Classification/ # Ionosphere Dataset (KNN Classification)
├── Chapter_03_NBA/            # NBA Sports Analytics (Decision Trees/Random Forests)
├── Chapter_04_Affinity/       # Affinity Analysis (Association Rules)
├── Chapter_05_Features/       # Adult Census (Feature Selection & PCA)
├── Chapter_06_Sentiment/      # Twitter Sentiment Analysis (Naive Bayes)
├── requirements.txt           # List of Python dependencies
└── README.md                  # This file
```

Each chapter folder contains:
*   **Source Code** (`.py` files): The actual implementation.
*   **Datasets**: Necessary `.data`, `.csv`, or `.txt` files.
*   **Concepts Guide** (`ChapterX_Concepts.md`): A detailed explanation of the theory and code.

## 🚀 Getting Started

Follow these instructions to set up your environment and run the code.

### 1. Prerequisites
*   **Python 3.10+** installed on your system.

### 2. Create a Virtual Environment (Recommended)
It is best practice to run Python projects in a virtual environment to avoid conflicts.

**Windows:**
```powershell
# Open command prompt or PowerShell in the project directory
python -m venv .venv
# Activate the environment
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Once the environment is active, install the required libraries:

```bash
pip install -r requirements.txt
```

This will install `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `tweepy`, and `nltk`.

## 🏃‍♂️ How to Run the Code

Since the code uses **relative paths**, you can run any script from the root directory or from within the chapter folder.

**Example: Running Chapter 4 (Affinity Analysis)**
```powershell
# From the root directory:
python Chapter_04_Affinity/ch4.py
```

**Example: Running Chapter 2 (Classification)**
```powershell
python Chapter_02_Classification/ch2.py
```
*(Note: Chapter 2 may open a plot window. Close it to finish the script)*

## 📚 Chapter Overviews

*   **Chapter 1**: Intro to Scikit-Learn.
*   **Chapter 2**: Predicting "Good" vs "Bad" radar signals in the Ionosphere dataset using K-Nearest Neighbors.
*   **Chapter 3**: Predicting NBA game winners using Decision Trees and Random Forests. FEATURE ENGINEERING involves creating "HomeLastWin" attributes.
*   **Chapter 4**: Affinity Analysis (Market Basket Analysis) finding rules like "If X, then Y" (e.g., Bread -> Milk).
*   **Chapter 5**: Feature Selection (Chi-Squared) and Dimensionality Reduction (PCA) on the Adult Census dataset.
*   **Chapter 6**: Sentiment Analysis on Tweets using Naive Bayes. Can classify text as "Positive" or "Negative".

---
*Created by [Your Name] for Data Mining Class Review.*
