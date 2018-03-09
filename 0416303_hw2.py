import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold
import random
import timeit
from sklearn.preprocessing import normalize
from sklearn import preprocessing

data = []
with open('winequality-white.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		data.append(row)

newData = []
for row in data:
	newData.append(row[0].split(';'))


data = np.array(newData)
data = data[np.argsort(data[:, 11])]
x, y = data[1:,0:10], data[1:,11]
x = x.astype(float)
y = y.astype(float)
knum = 5
#************************************************************
norm_x = preprocessing.normalize(x,axis=1)
norm_y = y
algorithm = ['brute','kd_tree']
metric = ['euclidean','manhattan','euclidean']
for j in range(6):
	resub_confusionMatrix = np.array([[0 for u in range(7)] for v in range(7)])
	kFold_confusionMatrix = np.array([[0 for u in range(7)] for v in range(7)])
	resub_accuracy_score = 0
	kFold_accuracy_score = 0
	time = 0
	if j%3 == 2:
		met = "cosine"
		data_x = norm_x
		data_y = norm_y
	else:
		met = metric[j%3]
		data_x = x
		data_y = y
	for i in range(knum):
		start = timeit.default_timer()
		kf = StratifiedKFold(n_splits=knum, random_state=i*random.randrange(10), shuffle=True)
		for train, test in kf.split(data_x,data_y):
			x_train, x_test, y_train, y_test = data_x[train], data_x[test], data_y[train], data_y[test]
		neigh = KNeighborsClassifier(algorithm=algorithm[j%2], n_neighbors=30, metric=metric[j%3], weights='distance')
		resub_clf = neigh.fit(data_x, data_y)
		kFold_clf = neigh.fit(x_train, y_train) 
		stop = timeit.default_timer()
		time += stop - start
		resub_confusionMatrix += confusion_matrix(data_y,resub_clf.predict(data_x))
		kFold_confusionMatrix += confusion_matrix(y_test,kFold_clf.predict(x_test))
		resub_accuracy_score += accuracy_score(data_y, resub_clf.predict(data_x))
		kFold_accuracy_score += accuracy_score(y_test, kFold_clf.predict(x_test))
	time /= knum
	resub_confusionMatrix /= knum
	kFold_confusionMatrix /= knum
	resub_accuracy_score /= knum
	kFold_accuracy_score /= knum
	print "********************" + algorithm[j%2] + " & " + met + "********************"
	print "Time:"
	print time
	print "******* Resubstitution Result *******"
	print "Confusion Matrix:"
	print resub_confusionMatrix
	print "Accuracy Score:"
	print resub_accuracy_score
	print "****** 10-Fold Result ******"
	print "Confusion Matrix:"
	print kFold_confusionMatrix
	print "Accuracy Score:"
	print kFold_accuracy_score


