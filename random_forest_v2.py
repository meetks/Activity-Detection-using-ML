#!/usr/bin/python -tt 

import sys, getopt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import csv 
import numpy as np
import scipy
from scipy import stats
import time
from sklearn import svm

training_files = []
testing_files = []

class activity_classifier:
  def __init__(self):
    self.default_classifier = RandomForestClassifier(n_jobs=10)
    print 'RandomForest Is the default classifier'

  def __init__(self,c):
    if (c == 'A' or c == 'a') :
      self.default_classifier = AdaBoostClassifier( DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5, algorithm="SAMME")
      print 'AdaClassifier Is the default classifier'
    if (c == 'R' or c == 'r') :
      self.default_classifier = RandomForestClassifier(n_jobs=10)
      print 'RandomForest Is the default classifier'
    if (c == 'S' or c == 's') :
      self.default_classifier = svm.SVC(kernel='linear')
      print 'SVM Is the default classifier'
         

  def set_classifier(self, name):
    self.default_classifier = name

  def get_classifier(self):
    return(self.default_classifier)

  def fit(self, inp, labels):
    return ((self.get_classifier()).fit(inp, labels))

  def predict(self, test):
    return((self.get_classifier()).predict(test))
    
def update_counter(p, actual, err_counters):
  err_counters[p][actual] += 1
  

def find_energy(X):
# Based of Parseval's theorem 
  y = np.fft.fft(X)
  return(np.sum(np.abs(y) ** 2) / len(X))

def find_features(arr, features):
   j = 0 
   temp = []
   while j < 3 :
      mean = np.mean(arr[j])  
      sd = np.std(arr[j])  
      energy = find_energy(arr[j])
      kurtosis = scipy.stats.mstats.kurtosis(arr[j])
      temp.extend([mean,sd,energy,kurtosis])
      j = j + 1
   # Calculate correlation between X & Y
   xycorr = np.corrcoef(arr[0], arr[1])
   zycorr = np.corrcoef(arr[2], arr[1])
   zxcorr = np.corrcoef(arr[2], arr[0])
   temp.extend([xycorr[0,1], zycorr[0,1], zxcorr[0, 1]])
   features.append(temp)
   


def extract_features(inp, label):
  i = 0
  features = []
  feat_label = []
  last_label = 1 
  window_size = 0
  count = 0
  while i < len(inp) :
    while (i < len(inp) and last_label == label[i]) : 
      last_label = label[i]
      i = i + 1
      window_size = window_size + 1 
      if (window_size % 10 == 0) :
         break

    temp = inp[i:i+window_size]
    if not temp:
      break
    transpose = zip(*temp) 
    find_features(transpose, features)
    feat_label.append(last_label)
    # Append all values and corr of x & y
    last_label = label[i]
    window_size = 0
    count = count + 1 
  return features, feat_label, count


def read_file_csv (datafiles, count, test_ornot):
    data = []
    label = []
    last_label = 1
    for datafile in datafiles:
        print "Using file:" + str(datafile)
        fp = open(datafile,"r");

        reader = csv.reader(fp,delimiter=',')
        for row in reader:
          count = count + 1
          if test_ornot == True and (count % 2 )== 0:
            print count
            continue
          if test_ornot == False and (count % 2 )== 1:
            continue
          temp = []
          label.append(int(row[4]))
          temp.append(int(row[1]))
          temp.append(int(row[2]))
          temp.append(int(row[3]))
          data.append(temp)
        print len(data), len(label)
    return data, label, count


def read_file(datafile, count):
    print "Using file:" + str(datafile)
    fp = open(datafile,"r");

    data = []
    label = []
    reader = fp.readlines()
    for row in reader:
      dataset = row.split(" ")
      temp = []
      label.append(int(dataset[2]))
      temp.append(dataset[1])
      data.append(temp)
      count = count + 1
    return data, label, count

def print_err_counters(arr, err):
  k = 0 
  j = 0
  while k < 8 :
    j = 0
    while j < 8 :
      err.write( '%5d' % (arr[k][j]))
      j += 1
    err.write( '\n')
    k += 1
     
def test_model(classifier, test_vector, test_count, test_label, err):
   count = 0 
   correct = wrong = 0
   err_counters= [[0]*8 for i in range(8)]
   while count < len(test_vector):
     try:
       p = classifier.predict(test_vector[count])
     except ValueError:
         err.write("Problem with %d" % count);
         count = count + 1 
         continue;
     count = count + 1 
     if count >= test_count :
       break
     if int(p) == test_label[count]:
       correct = correct + 1
     else :
       update_counter(int(p), test_label[count], err_counters)
       wrong = wrong + 1
       err.write(str(int(p))+" " + str(test_label[count])+" " + str(count) + "\n")
   print wrong, correct, count
   accuracy = float(float(correct)/count) * 100.0 
   print accuracy
   print_err_counters(err_counters, err)

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"ht:c:e:",["trainfile=","testfile=","errorfile="])
   except getopt.GetoptError:
      print 'random_forest.py -t <trainfile> -c <testfile> -e <errorfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'random_forest.py -t <trainfile> -c <testfile> -e <errorfile>'
         sys.exit()
      elif opt in ("-t", "--trainfile"):
         training_files.append(arg)
         print 'Training files %s' % training_files
      elif opt in ("-c", "--testfile"):
         testing_files.append(arg)
         print 'Testing files %s' % testing_files
      elif opt in ("-e", "--errorfile"):
         err_file = arg
         print 'Error log %s' % err_file
   err  = open(err_file,"w")
   count_train = 0
   count_test = 0
   #data, label, count = read_file(training_files[0], count)
   data, label, count_train = read_file_csv(training_files, count_train, False)
   datatest, label_test, count_test = read_file_csv(testing_files, count_test, True)
   
   features_train, feature_train_label, feat_train_count = extract_features(data, label)
   features_test, feat_test_label, feat_test_count = extract_features(datatest, label_test)
   act_class = activity_classifier('S')
   act_class.fit(features_train, feature_train_label) 
   tick_start = time.time()
   test_model(act_class, features_test, feat_test_count, feat_test_label, err)
   tick_stop = time.time()
   print 'Time taken %f' % (tick_stop - tick_start)

if __name__=='__main__':                                                                                 
  main(sys.argv[1:])                                                                                                 
