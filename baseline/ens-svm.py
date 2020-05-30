from sklearn import tree
from sklearn import svm
import numpy as np


train_data = np.load('../course_train.npy')
#test_data = np.load('../test_data.npy')

def get_train_data(context):	
	x = []
	y = []
	test = []
	test_label = []
	test_context = train_data[0][-2]
	for line in train_data:
		if line[-2] == test_context:
			test.append(line[:-2])
			test_label.append(line[-1])
		elif line[-2]%3 == context:
			x.append(line[:-2])
			y.append(line[-1])
	return x, y

def get_test_data():
	x = []
	y = []
	test_context = train_data[0][-2]
	for line in train_data:
		if line[-2] == test_context:
			x.append(line[:-2])
			y.append(line[-1])
	return x, y, test_context

def ensemble():
	test, test_label, test_context = get_test_data()
	pred = []
	pred_test = []
	for i in range(3):
		#if i != (test_context%3):
		train, label = get_train_data(i)
			#clf = tree.DecisionTreeClassifier()
		clf = svm.LinearSVC()
		clf = clf.fit(train, label)
		predi = clf.predict(test)
		pred.append(predi)
		acc = 0
		for j in range(len(test_label)):
			if test_label[j]==predi[j]:
				acc += 1
		print("train " + str(i) + " acc:", acc/len(test_label))
	for i in range(len(pred[0])):
		l = [0]*10
		for j in range(len(pred)):
			l[int(pred[j][i])] += 1
		m = 0
		p = 0
		for k in range(10):
			if l[k] > m:
				p = k
				m = l[k]
		pred_test.append(p)
	acc_t = 0
	for i in range(len(test_label)):
		if test_label[i]==pred_test[i]:
			acc_t += 1
	print("ensemble acc:", acc_t/len(test_label) )

if __name__ == "__main__":
	ensemble()
