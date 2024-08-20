from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import nltk

# Download necessary NLTK datasets for text processing
nltk.download('stopwords')
nltk.download('punkt')

# Path to the dataset
DATASET_PATH = '2cls_spam_text_cls.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(DATASET_PATH)
print(df)

# Extract messages and labels from the DataFrame
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

# Label encoding: Convert categorical labels into numerical labels
le = LabelEncoder()
# Encode labels (e.g., 'spam' and 'ham' to 1 and 0)
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')

# Functions for text preprocessing


def lowercase(text):
    """Convert all characters in the text to lowercase."""
    return text.lower()


def punctuation_removal(text):
    """Remove all punctuation from the text."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def tokenize(text):
    """Tokenize the text into individual words."""
    return nltk.word_tokenize(text)


def remove_stopwords(tokens):
    """Remove common stopwords (e.g., 'and', 'the') from the token list."""
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in tokens if token not in stop_words]


def stemming(tokens):
    """Apply stemming to reduce words to their root form (e.g., 'running' -> 'run')."""
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text):
    """Apply a series of preprocessing steps to the text: lowercasing, punctuation removal,
    tokenization, stopword removal, and stemming."""
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens


# Apply preprocessing to all messages
messages = [preprocess_text(message) for message in messages]


def create_dictionary(messages):
    """Create a dictionary of unique words from the list of tokenized messages."""
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)
    return dictionary


def create_features(tokens, dictionary):
    """Convert a list of tokens into a feature vector based on the dictionary."""
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features


# Build the dictionary and feature vectors for all messages
dictionary = create_dictionary(messages)
X = np.array([create_features(tokens, dictionary) for tokens in messages])

# Split the dataset into training, validation, and test sets
VAL_SIZE = 0.2  # 20% of the data will be used for validation
TEST_SIZE = 0.125  # 12.5% of the data (from training) will be used for testing
SEED = 0  # Seed for reproducibility

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()
print('Start training...')

# Train the model using the training data
model = model.fit(X_train, y_train)
print('Training completed!')

# Evaluate the model's performance on validation and test sets
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')

# Function to make predictions on new text input


def predict(text, model, dictionary):
    """Predict the class of a given text input using the trained model."""
    processed_text = preprocess_text(text)  # Preprocess the input text
    features = create_features(
        processed_text, dictionary)  # Create feature vector
    features = np.array(features).reshape(
        1, -1)  # Reshape to match model input
    prediction = model.predict(features)  # Predict the label
    # Convert label back to original class name
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls


# Test case: Predict the class of a sample text input
test_input = 'I am Hieu'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')
