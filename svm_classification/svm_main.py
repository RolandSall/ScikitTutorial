from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
firstFeature = iris.data
secondFeature = iris.target
classes = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

x_train, x_test, y_train, y_test = train_test_split(firstFeature, secondFeature, test_size=0.4)
model = svm.SVC()
model.fit(x_train, y_train)
# print(model)

prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)

print("Tested Value Actual value: ", prediction)
print("Predicted Value", y_test)
print("Accuracy", accuracy)
