## Credit Risk Assessment

This is a Python-based Credit Risk Assessment project that uses a Random Forest Classifier to predict the loan status of individuals based on various features related to credit risk. The dataset is assumed to be in CSV format with columns containing different credit-related attributes, including the loan status.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [How it Works](#how-it-works)
- [Usage](#usage)

## Dataset

The dataset file credit_risk_dataset.csv contains various credit-related attributes, including the loan status. The CSV format should have columns for these attributes.

You can download my dataset directly from this link:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset/download?datasetVersionNumber=1

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- pandas
- scikit-learn
- matplotlib

You can install the required packages using the following command:

pip install pandas scikit-learn matplotlib

## How it Works

The Credit Risk Assessment works as follows:

1- The dataset (CSV format) is loaded, containing various credit-related attributes, including the loan status.

2- Categorical variables are converted to numeric using LabelEncoder.

3- The data is split into features (X) and target (y), where the target is the loan status.

4- Missing values are handled using SimpleImputer, where NaN values are replaced with the mean.

5- The imputed data is split into training and testing sets using an 80-20 split.

6- A Random Forest Classifier model is created and trained on the training data.

7- Predictions are made on the test data, and the model's performance is evaluated using accuracy, confusion matrix, and classification report.

8- Feature importances are extracted from the trained model and plotted to show the importance of each feature.

## Usage

1- Clone the repository or download the creditriskassesment.py and credit_risk_dataset.csv files.

2- Make sure you have Python 3.x installed on your system.

3- Install the required dependencies by running pip install pandas scikit-learn matplotlib.

4- Run the creditriskassesment.py script.

The script will load the dataset, preprocess the data, train the Random Forest Classifier, and assess the credit risk of individuals. Additionally, it will display the model's performance and plot the feature importances.
