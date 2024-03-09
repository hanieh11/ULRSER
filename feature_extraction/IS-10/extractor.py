import os

files = os.listdir('wav')

for file in files:
    os.system("./SMILExtract -C ../../config/IS10_paraling.conf -I wav/" + file + " -O features/" + file[:-4] + ".arff")

