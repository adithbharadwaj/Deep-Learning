from sklearn import datasets # importing the dataset
from sklearn import svm # importing support vector machine

import matplotlib.pyplot as plt
# plotting the data

#loading the digits
digits = datasets.load_digits() 

gamma_val = 0.001 # value of gamma
c_val = 100 # value of c

# creating a support vector classifier from svm.
clf = svm.SVC(gamma = gamma_val, C = c_val)

#splitting the data into traning samples.
x_train, y_train = digits.data[:-100], digits.target[:-100]

#fitting the data
clf.fit(x_train, y_train)

#predicting on a sample
print(clf.predict(digits.data[-4].reshape(1, -1)))

# displaying the image of that sample.
plt.imshow(digits.images[-4], cmap = 'gray')
plt.show()