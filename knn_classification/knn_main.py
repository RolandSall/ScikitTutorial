import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PREDICTED_CAR = 32

pd.options.mode.chained_assignment = None
carData = pd.read_csv('car.data')
# To view the data, caraData.head() print the first 5 rows
# print(carData)

features = carData[['buying', 'maint', 'safety']].values
label = carData[['class']]
# print(features, label)

# convert strings into numbers for ML

labelEncoder = LabelEncoder()
for index in range(len(features[0])):
    features[:, index] = labelEncoder.fit_transform(features[:, index])

# print(features)

# built in mapper

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3,
}
label['class'] = label['class'].map(label_mapping)
label = np.array(label)
# print(label)

# creating model

knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# train
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.7)
knn_classifier.fit(x_train, y_train)

prediction = knn_classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("Prediction", prediction)
print("Accuracy", accuracy)


print("Tested Value Actual value: ", label[PREDICTED_CAR])
print("Predicted Value", knn_classifier.predict(features)[PREDICTED_CAR])
