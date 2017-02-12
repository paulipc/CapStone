import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import timeit

print 'Reading files and making predictions...'
start = timeit.default_timer()

# read file into dataframe
df = pd.read_csv('./inputs/F1_All_Traffic.csv')

# need to read whatever file in the directory and then delete them
# requirement for "service"

# filter all 'Denied' lines
df2 = df[df.msg != 'Denied']

# split 'info_5' field information into substrings
# expand dataframe with 0,1,2 columns
# where 1 = sent, 2 = received
df3 = pd.concat([df2, df2['info_5'].str.split(";", expand=True)], axis=1, join_axes=[df2.index])

# create new field for sent and received data

# move traffic information to 'sent' and 'received' fields
# cleans text and leaves only numbers
df3['received'] = pd.to_numeric(df3[2].str.replace('rcvd_bytes=', ''))
df3['sent'] = pd.to_numeric(df3[1].str.replace('sent_bytes=', ''))
df3['date'] = pd.to_datetime(df3['update_time'])

# select needed fields only to new dataframe
df4 = df3[['date', 'sent', 'received']]

####
# this makes dataframe to sum data traffic to 15 min. periods
###
# set update time to index
df4 = df4.set_index(['date'])
# resample, sum rolling and mean gives us needed information
df5 = df4.resample('15T').sum().fillna(0).rolling(window=15, min_periods=1).mean()
# df5['date2'] = df5['date']
df5.to_csv('./outputs/output.csv')

# it's challenging to convert rowmatrix to numpy.matrix
# workaround is to write temporary file and load file to another format
str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

A = np.genfromtxt('./outputs/output.csv', dtype=None, names=True, delimiter=',', converters={0: str2date})

X_temp = np.asarray(zip(*A)[0], dtype=datetime)
y = np.asarray(zip(*A)[1], dtype="float64")

# date.toordinal() gives number of date as int, make this big int and add hours and minutes
# then we get X
X_temp2 = np.zeros(shape=(len(X_temp),), dtype="float64")

for i in range(0, len(X_temp)):
    X_temp2[i] = date.toordinal(X_temp[i]) + float(X_temp[i].hour) / 100 + float(X_temp[i].minute) / 10000

# this is needed, otherwise fit fails
X = X_temp2.reshape((len(X_temp2), 1))

# KNeighbors Regressor
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)

# print("Test set predictions:\n{:.4f}".format(reg.predict(X_test)))
# this code is for printing only
# X_pred = reg.predict(X_test)
# for i in range(0,len(X_pred)):
#     print("%.4f" % X_pred[i])
predictions = reg.predict(X_test)
# accuracy = reg.score(X_test, y_test)
np.savetxt('./outputs/predictions.csv', predictions, delimiter=',')
# np.savetxt('./outputs/accuracy.csv', accuracy, delimiter=',')
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))
stop = timeit.default_timer()

print 'time: ', stop - start, ' sec.'

# TODO split functionality to two different files: train and predict
# TODO from inputfile remove weekends and public holidays
# TODO make a graph to see how predictions look
# TODO check next things on the list if this now is soon ok
# TODO check how input can variate and if some extra logic needs to be done regarding this
