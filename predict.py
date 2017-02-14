import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from datetime import date
import csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import timeit

start = timeit.default_timer()
print 'time: ', start, ' sec.'

filename = './outputs/trained_model.sav'
try:
    reg = pickle.load(open(filename, 'rb'))
except:
    print 'cannot open file or directory'
    print 'program exits'
    raise

X_test = np.load('./outputs/X_test.npy')
y_test = np.load('./outputs/y_test.npy')
#Y_test = np.load('./outputs/Y_test')

predictions = reg.predict(X_test)

# accuracy = reg.score(X_test, y_test)
np.savetxt('./outputs/predictions.csv', predictions, delimiter=',')
# np.savetxt('./outputs/accuracy.csv', accuracy, delimiter=',')

print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

stop = timeit.default_timer()
print 'time: ', stop - start, ' sec.'

# TODO make a graph to see how predictions look
# TODO check if both sent and received traffic needs to be included and summed, or how?
