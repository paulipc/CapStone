#
#
#   USAGE: python mgrep.py firststring secondstring
#
#   Example: python mgrep.py #TODO 2017.02
#
#   Desc: prints files containing both strings on same line or not in same line
#
#   Note: run from local directory i.e. "./"
#           this is dev version so runs happy path
#
#
#!!!C:\Python27\python.exe
from os import listdir
from os.path import isfile, join
import sys

#mypath = 'C:/Users/pauli.sutelainen/Documents/Private/test'
mypath = './'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# files contains list of files in directory

txtfiles = list()
for f in files:
    if '.txt' in f:
        txtfiles.append(f)

searchstring1 = sys.argv[1]
searchstring2 = sys.argv[2]

ssl1 = list() # list for search string 1 containing files
for myfile in txtfiles:
    filename = mypath+'/'+myfile

    file = open(filename,"r")
    for line in file:
        if searchstring1 in line:
            ssl1.append(myfile)
            file.close()
            break
    file.close()

ssl2 = list() # list for search string 1 containing files
for myfile in txtfiles:
    filename = mypath+'/'+myfile

    file = open(filename,"r")
    for line in file:
        if searchstring2 in line:
            ssl2.append(myfile)
            file.close()
            break
    file.close()

# now we have ssl1 and ssl2 two list that contains the strings

for i in set(ssl1).intersection(ssl2):
    with open(i) as f:
        for line in f:
            if searchstring1 in line or searchstring2 in line:
                line = i+': '+line
                sys.stdout.write(line)
