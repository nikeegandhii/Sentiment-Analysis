# Sentiment-Analysis
This GitHub project is a machine learning solution for sentiment analysis of IMDb movie reviews. Sentiment analysis involves classifying movie reviews as either "positive" or "negative" based on their content. Here's a summary of what this project does:

**Key Components**:
1. **Data Loading**: It loads the IMDb movie reviews dataset from a CSV file. (Data was downloaded from Kaggle for practice)
2. **Data Preprocessing**: The text data is prepared for modeling, and sentiment labels are transformed into binary values.
3. **Data Splitting**: The dataset is divided into training, validation, and testing sets.
4. **Tokenization and Padding**: Text data is converted into numerical sequences and padded to ensure uniform input size.

5. **Model Building and Training**:
- The core of this project is a deep learning model that includes an embedding layer, an LSTM layer, and a dense layer for binary classification.
- The model is trained on the training data and evaluated using the validation set.
- Training progress is visualized with a plot showing loss over epochs.

6. **Model Evaluation**:
- The model's performance is assessed on the testing data using accuracy, a confusion matrix, and a classification report.

Data: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
