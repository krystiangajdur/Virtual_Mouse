import joblib
import numpy as np
from sklearn.svm import SVC

X = np.load('x_data.npy')
Y = np.load('y_data.npy')

model = SVC(kernel = 'linear')

model.fit(X,Y)
print('The model has been trained.')

joblib.dump(model, 'svm_hand_model.pkl')
print('The model has been saved as "svm_hand_model.pkl". ')


