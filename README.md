# Sentiment Analysis of Twitter Data

## Project Overview
This project performs sentiment analysis on Twitter data using a machine learning model. The dataset used is the [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains millions of tweets labeled as either positive or negative. The model is built using a Logistic Regression classifier, and the project allows users to input text (tweets) and get real-time sentiment predictions. The project also includes a web app for easy interaction and visualization.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone behind a body of text. In this project, we analyze tweets to understand public sentiment on various topics. By classifying tweets into positive and negative sentiments, we can gain valuable insights into public opinion, which is helpful for businesses, policymakers, and researchers.

Twitter is a platform where users express thoughts, opinions, and reactions to various subjects. Analyzing this data is essential for businesses seeking to understand customer feedback or gauge public reaction to events, products, and services.

This project leverages the Sentiment140 dataset, which contains a large number of tweets labeled as positive or negative. Using this labeled dataset, we train a Logistic Regression model to predict the sentiment of unseen tweets. The model incorporates text preprocessing, feature extraction, and evaluation metrics to ensure high performance and accuracy.

## Dataset
The dataset used for this project is the Sentiment140 Dataset, which consists of 1.6 million tweets labeled with sentiments (positive or negative). The dataset includes:
- Target: Sentiment label (0 = negative, 4 = positive)
- Text: The tweet content
- Other Columns: Metadata (e.g., tweet id, date, user)

The dataset is preprocessed by:
- Replacing target labels (0 = negative, 4 = positive)
- Removing irrelevant metadata (such as user information, date, etc.)
- Text preprocessing (e.g., stemming, stopwords removal)

## Installation
- Clone this repository or download the project files.
- Install the required dependencies by running:
```
pip install -r requirements.txt
```
requirements.txt file:
```
# Core dependencies
numpy
pandas
scikit-learn
nltk

# Data extraction
zipfile36  # For handling zip files

# Model training and evaluation
matplotlib
seaborn

# Streamlit for the web app
streamlit
```
- Set up your Kaggle API credentials by downloading the kaggle.json file from your Kaggle account and placing it in the directory ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows).
- Download the dataset using the following command:
```
kaggle datasets download kazanova/sentiment140
```

## Usage
- Download and Prepare the Dataset
Once the dataset is downloaded, extract the files into the sentiment140_extracted/ folder.

- Preprocess and Train the Model
After the data is prepared, the model is trained using the following steps:

1. Load the dataset and clean it.
2. Preprocess the text by removing non-alphabetical characters and applying stemming.
3. Split the data into training and testing sets (80% for training, 20% for testing).
4. Convert the text into numerical vectors using the TF-IDF Vectorizer.
5. Train the Logistic Regression model on the preprocessed data.
6. Save the trained model and vectorizer using pickle for future use.

- Run the Streamlit Web App
Once the model is trained, you can start the web app by running the following command:
```
streamlit run app.py
```
This will open a web interface in your browser where you can enter a tweet, and the app will predict whether the sentiment is positive or negative.

- Input and Prediction
The web app provides a text input box where users can enter any tweet. Upon clicking the Analyze Sentiment button, the app will display the predicted sentiment as either Positive or Negative based on the model’s output.

## Model
The sentiment analysis model is built using Logistic Regression, a classification algorithm suitable for binary classification tasks. The following steps outline the model-building process:

- Text Preprocessing:
1. Cleaning: Removing non-alphabetical characters.
2. Stopwords Removal: Filtering out common words that don't contribute to sentiment.
3. Stemming: Reducing words to their root form using the Porter Stemmer.

- Feature Extraction:
1. TF-IDF Vectorizer: Converts the text data into a numerical format (word frequencies) that the machine learning model can interpret.

- Model Training:
The model is trained using the preprocessed data. Logistic Regression is used as it is effective for binary classification tasks and performs well with high-dimensional data (e.g., text data).

- Evaluation:
The model’s performance is evaluated on both the training and test datasets. The model achieved over 75% accuracy, indicating reliable performance on unseen data.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
