import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset has 15 classes (dialog acts)
    
# We want to read the dalog_acts.dat file. The first word of every line is the class label and the rest of the line is the text.

df = pd.read_csv('C:/Users/nikol/OneDrive/Desktop/UU/Period-1/Methods-in-AI/MAIR-30/TeamProject/Part1-SystemImplementation/Part1a-TextClassification/dialog_acts.dat', header=None, names=['data'])

# Apply the function to split the 'data' column into 'label' and 'text' columns
df[['label', 'text']] = df['data'].apply(lambda x: pd.Series(x.split(' ', 1)))

# Drop the original 'data' column
df.drop('data', axis=1, inplace=True)

# print(df.head(10))

# Features and Labels
x = df['text']
y = df['label']

# Splitting the dataset into training and test sets
# 85% of the data is used for training and 15% for testing
# random state is like seed I think to just keep the same split and shuffling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=10, shuffle=True)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# print(x_train.head(85))
# print(y_train.head(15))
# print(x_test.head(85))
# print(y_test.head(15))

####################### Baseline majority class (inform label) #######################
# Identify the majority class (idk if its needed)
# majority_class = y_train.value_counts().idxmax()

def baseline_majority(y_test, majority_class='inform'):
    total_instances = len(y_test)
    correct_predictions = (y_test == majority_class).sum()
    
    accuracy = (correct_predictions / total_instances) * 100
    return f"{accuracy:.2f}%"

baseline_majority_accuracy = baseline_majority(y_test)
print(f"Baseline majority accuracy: {baseline_majority_accuracy}")
####################### Baseline majority class (inform label) #######################

############################## Baseline keyword matching #############################



############################## Baseline keyword matching #############################