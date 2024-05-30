# Hi, I'm Will Wang, this is intro page for the Credit_Card_Fraud machine learning model! 
<p>Software Enginner at <a href="https://www.engineering.utoronto.ca/">University of Toronto</a></br></p>

### A little more about me and the algorithm...  

```javascript
const Credit_Card_Fraud = {
  name: "Credit_Card_Fraud",
  code: [Python],
  tools: [Git, Notebook],
  take_away: "Give it a try"
}
```

## Introduction
Hello, I'm Will Wang, a Software Engineer from the University of Toronto. This project is part of my exploration into using machine learning to combat credit card fraud, a significant concern for both businesses and consumers worldwide. The Credit_Card_Fraud Detection Model utilizes logistic regression to identify fraudulent transactions based on encoded transaction data.

## Project Overview
The convenience of credit cards comes with the risk of fraud. This project aims to detect such fraudulent activities using a logistic regression algorithm applied to a dataset of credit card transactions.

### Dataset
The dataset consists of 284,808 credit card transactions with features that are reduced through PCA to 28 principal components (V1 to V28) to maintain confidentiality. Each transaction is labeled as fraudulent or not ('Class' variable), providing a clear target for our predictive model.
<img width="1646" alt="截屏2023-03-29 17 11 56" src="https://user-images.githubusercontent.com/105031962/228668634-fb5e6816-e48a-4068-889a-a29f8f1aa7b5.png">

### Features
- **Encoded Features (V1-V28)**: PCA-transformed attributes of transactions, ensuring anonymity.
- **Amount**: Transaction amount, this feature is not transformed.
- **Class**: Binary variable indicating fraudulent (1) and non-fraudulent (0) transactions.

## Setup
### Prerequisites
- Python 3.x
- Pandas
- Matplotlib
- NumPy
- Scikit-Learn

### Installation
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/your-repository/Credit_Card_Fraud.git
cd Credit_Card_Fraud
pip install -r requirements.txt
```

## Usage
Run the main script to train the model and evaluate its performance on the dataset:
```bash
python credit_card_fraud_detection.py
```

## Methodology

### Data Preprocessing
- **Normalization**: We scale the 'Amount' feature to have zero mean and unit variance using standard scalar normalization. This step is crucial as it brings all input features to the same scale, allowing the model to converge more quickly during training.
- **Splitting Data**: The dataset is split into training (70%) and testing (30%) sets. This separation ensures that we have a reliable way to evaluate the model's performance on unseen data, simulating how it would perform in a real-world scenario.

### Model Building
- **Logistic Regression**: Logistic regression is chosen for its efficiency and effectiveness in binary classification tasks. Its interpretability, despite the use of encoded features, also makes it a preferred choice. It models the probability of class membership as a logistic function of the predictors.
- **Under-sampling**: Due to the imbalance between fraudulent and non-fraudulent transactions in the dataset, we perform under-sampling of the majority class. This technique involves randomly selecting a subset of non-fraudulent transactions equal in size to the fraudulent transactions, which helps to prevent the model from being biased toward the more common class.

### Evaluation
- **Confusion Matrix**: A confusion matrix is used to visualize the performance of the classification model. Each entry in the matrix allows us to infer not just the overall accuracy but also the specific types of errors (type I and type II).
- **Precision-Recall**: These metrics are particularly important in the context of fraud detection, where the cost of false negatives (failing to detect fraud) can be very high. Precision measures the accuracy of the positive predictions, and recall measures the ability of the model to find all the positive instances (actual frauds).

## Contributing
We welcome contributions from the community. Whether it's fixing bugs, improving the documentation, or adding new features, your help is appreciated. Please follow the steps below to contribute:

1. **Fork the Repository**: Start by forking the repository to your GitHub account.
2. **Create a Feature Branch**: Make a new branch for your feature or fix. This keeps the main branch free from potential unstable changes.
   ```bash
   git checkout -b feature/YourAmazingFeature
