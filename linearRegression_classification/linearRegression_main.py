from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

bostonDataSet = datasets.load_boston()

features = bostonDataSet.data
label = bostonDataSet.target

linearReg = linear_model.LinearRegression()

plt.scatter(features.T[0], label)

plt.show()

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
model = linearReg.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Prediction", prediction)

