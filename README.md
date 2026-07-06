# IoMT Machine Learning Benchmark

## Project Overview
This project is an automated machine learning benchmarking tool. It evaluates the performance of 7 standard classification algorithms to determine which model is best suited for detecting malicious network anomalies and cyber attacks (such as ARP Spoofing) within Internet of Medical Things (IoMT) environments.

## Methodology
The pipeline follows standard machine learning engineering practices:

1. **Data Generation:** Because a raw dataset was not provided, the script uses `sklearn` to engineer a synthetic dataset of 1,000 patient records, featuring 10 numerical sensor readings and a binary target label (Healthy vs. Sick).
2. **Preprocessing:** The data is split into an 80% Training Set and a 20% Testing Set to prevent overfitting. It utilizes `SimpleImputer` to handle missing values and `StandardScaler` to ensure all sensor readings are evaluated on an equal mathematical scale.
3. **Model Training:** The script instantiates and trains 7 different classification models simultaneously on the 80% training data:
   * Logistic Regression
   * Decision Tree
   * Random Forest
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Gaussian Naive Bayes
   * AdaBoost
4. **Evaluation:** The trained models are tested against the 20% unseen test data. The script calculates Accuracy, Precision, Recall, and the F1-Score for each algorithm.

## Repository Contents
* **`IoMT_ML_Benchmark.ipynb`**: The main Python script/notebook containing the data generation, training loop, and evaluation logic.
* **`synthetic_dataset.csv`**: The 1,000-row generated dataset used to train the models.
* **`ML_Model_Comparison.csv`**: The final output report containing the performance metrics of all 7 models. 

## Results
The script successfully exports the classification reports to a CSV file. In this specific run, ensemble methods performed the best, with the **Random Forest Classifier** achieving the highest overall accuracy (89.5%), followed closely by the Decision Tree (86.5%).
