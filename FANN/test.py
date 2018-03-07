from keras.utils import to_categorical

__author__ = 'bohaohan'
from Fann_Model import *
from FC import *
from Sigmoid import *



from sklearn.datasets import make_moons
from sklearn.cross_validation import train_test_split

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
y = to_categorical(y, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print X.shape
print y.shape
# print y

model = Fann_Model(X.shape)
model.add(FC(5))
model.add(Sigmoid())
model.add(FC(2))
model.add(Sigmoid())

model.fit(X_train, y_train, max_epoch=200, lr=0.01)

print model.evaluate(X_test, y_test)