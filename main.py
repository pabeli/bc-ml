# Import important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# path to dataset
filename="data/breast-cancer-wisconsin.data"

#set columns name
columns_name = [
    'sample_code',
    'clump_thickness',
    'uniformity_cell_size',
    'uniformity_cell_shape',
    'marginal_adhesion',
    'single_epithelial_cell_size',
    'bare_nuclei',
    'bland_chromatin',
    'normal_nucleoli',
    'mitoses',
    'class'
]

# load data
data = pd.read_csv(filename, names = columns_name)


# Replace "?" with "-9999"
data = data.replace('?', -9999)

# Define X (independent variables) and y (target variable)

X = data.drop(['class'], axis=1)

y = data['class']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# call our classifier and fit to our data
classifier = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=25,
    p=1,
    metric="minkowski",
    n_jobs=-1
)

# training the classifier
classifier.fit(X_train, y_train)

# test our classifier
result = classifier.score(X_test, y_test)

print(f'Accuracy is {result}')

model_name = "KNN_classifier"
joblib.dump(classifier, 'models/{}.pkl'.format(model_name))