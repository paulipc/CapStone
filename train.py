import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import timeit
import pickle
from os import listdir
from os.path import isfile, join
import sys
#
# This code reads input file from ./inputs -directory and save trained model to ./output -directory
#
print 'Reading file and training the model. Model will be stored to ./output directory.'
start = timeit.default_timer()

# find files in ./inputs directory
# only one file is allowed, this program uses exactly one input file

mypath = './inputs'
try:
    inputfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
except:
    print 'cannot open the directory'
    print 'program exits'
    raise

if len(inputfiles) == 1:
    pass
else:
    print 'this program takes exactly one input file in directory ./inputs'
    print 'program exits'
    sys.exit(1)

filename = ''
filename = str(inputfiles[0])

# read file into dataframe
df = pd.read_csv('./inputs/'+filename)

stop = timeit.default_timer()
print 'File read in time: ', stop - start, ' sec.'

# need to read whatever file in the directory and then delete them
# requirement for "service"

# filter all 'Denied' lines
df2 = df[df.msg != 'Denied']

# TODO: next thing is to move out weekends by date from dataframe i.e. sat=day nr.5, sun=day nr.6
# which can be done with In [11]: df2[df2.index.dayofweek < 5]

# TODO: next thing is to move bank holidays



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

# save the model to disk

filename = './outputs/trained_model.sav'
try:
    pickle.dump(reg, open(filename, 'wb'))
except:
    print 'cannot save model to file'
    print 'program exits'
    raise

filename = './outputs/y_test'
try:
    np.save('./outputs/y_test', y_test)
except:
    print 'cannot save y_test to file'
    print 'program exits'
    raise

filename = './outputs/X_test'
try:
    np.save('./outputs/X_test', X_test)
except:
    print 'cannot save X_test to file'
    print 'program exits'
    raise


#np.save('./outputs/Y_test',y_test) for some reason this is not needed.

print 'Model trained in time: ', stop - start, ' sec.'
print 'Trained model saved to ./outputs directory with filename trained_model.sav'
print 'Trained y_test saved to ./outputs directory with filename y_test.npy'
print 'Trained X_test saved to ./outputs directory with filename X_test.npy'
