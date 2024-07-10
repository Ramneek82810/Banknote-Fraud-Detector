import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=1)
model = GaussianNB()

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Separate data into training and testing groups
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# encode labels as numerical value
label_encoder = LabelEncoder()
y_training = label_encoder.fit_transform([row["label"] for row in training])
y_testing = label_encoder.transform([row["label"] for row in testing])

# Train model on training set
X_training = [row["evidence"] for row in training]
model.fit(X_training, y_training)

# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
predictions = model.predict(X_testing)

# convert predictions back to original labels
predicted_labels = label_encoder.inverse_transform(predictions)
actual_labels = [row["label"]for row in testing]

# Compute how well we performed
correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(actual_labels, predicted_labels):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
