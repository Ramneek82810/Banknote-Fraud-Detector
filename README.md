# 🏛️ Banknote Fraud-Detection Using Gaussian Naive-Bayes

This project focuses on the detection of forged banknotes using machine learning. A **Gaussian Naive Bayes** classifier is employed to distinguish between real and fake currency based on features extracted from the images of banknotes. 

## 📊 Dataset Information
- **File**: `banknotes.csv`
- **Features**: 
  - Variance of Wavelet Transformed Image
  - Skewness of Wavelet Transformed Image   
  - Kurtosis of Wavelet Transformed Image
  - Entropy
- **Target**:          
  - `0`: Authentic  
  - `1`: Forged     
The dataset is sourced from the UCI Machine Learning Repository. 
 
## 🧠 Model Used
- **Algorithm**: Gaussian Naive Bayes 
- **Why Naive Bayes?**
  - Fast and simple
  - Works well with small datasets
  - Assumes feature independence, suitable for this dataset

## 🛠️ Files Overview
| File Name       | Description                                      |
|----------------|--------------------------------------------------|
| `banknotes.csv` | Dataset with real and fake banknotes            |
| `banknotes0.py` | Model training and prediction script (Version 1)|
| `banknotes1.py` | Improved/Alternate model implementation         |
| `README.md`     | Project documentation                           |

## 🚀 Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Ramneek82810/Banknote-Fraud-Detector.git
   cd Banknote-Fraud-Detector
   ```
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn
   ``` 
3. Run the script:
   ```bash
   python banknotes1.py
   ```

## 📈 Output
- Displays classification report and model accuracy.
- Predicts whether a given banknote is genuine or forged.

## ✅ Results
- Achieved high classification accuracy.
- Efficient performance on real-world-like data.

## 📌 Future Improvements
- Incorporate other classification algorithms (e.g., SVM, Random Forest).
- Build a web interface to allow users to upload data and get predictions.
- Visualize data distributions and correlations.
