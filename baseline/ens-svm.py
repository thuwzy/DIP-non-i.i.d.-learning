from sklearn import tree
from sklearn import svm
import numpy as np
import random
import sys

all_data = np.load('../course_train.npy')

def get_data():
	x = {}
	y = {}
	test_x = []
	test_y = []
	test_con = []
	train_con = []
	for i in range(10):
		train_con.append(i)
	for i in range(3):
		j = random.randint(0, 9-i)
		test_con.append(train_con[j])
		del train_con[j]
	for i in train_con:
		a = []
		b = []
		x[i] = a
		y[i] = b
	for line in all_data:
		if line[-2] in train_con:
			x[line[-2]].append(line[:-2])
			y[line[-2]].append(line[-1])
		elif line[-2] in test_con:
			test_x.append(line[:-2])
			test_y.append(line[-1])
		else:
			print(line)
	#print(len(x))
	return x, y, test_x, test_y


def ensemble(t_num):
	x, y, test, test_label = get_data()
	pred = []
	pred_test = []
	index = 0
	t_index = 0
	train = []
	label = []
	for i in x:
		index += 1
		t_index += 1
		train = train + x[i]
		label = label + y[i]
		if index==t_num or t_index==7:
			index = 0
			clf = svm.LinearSVC(max_iter=5000)
			clf = clf.fit(train, label)
			predi = clf.predict(test)
			pred.append(predi)
			acc = 0
			for j in range(len(test_label)):
				if test_label[j]==predi[j]:
					acc += 1
			print("train " + str(i) + " acc:", acc/len(test_label))
			train = []
			label = []

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
	return acc_t/len(test_label)

def pure_svm():
	data = np.load("../course_train.npy")
	X = data[:, :-2]
	Y = data[:, -1]
	C = data[:, -2]

	train_context_num = 7
	X_train = X[C < train_context_num]
	Y_train = Y[C < train_context_num]
	
	X_test = X[C >= train_context_num]
	Y_test = Y[C >= train_context_num]

	clf = svm.LinearSVC()
	clf = clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)

	acc = (pred == Y_test).sum() / len(Y_test)

	print('test acc:', acc)


if __name__ == "__main__":
	'''
	dc = {}
	dl = {}
	for line in all_data:
		if line[-1]==0:
			try:
				a = dc[line[-2]]
			except:
				print('context:', line[-2])
			dc[line[-2]] = True
		try:
			b = dl[line[-1]]
		except:
			print('label:', line[-1])
			dl[line[-1]] = True
	'''
	acc_ave = 0
	max_acc = 0
	min_acc = 1
	t_num = 0
	if len(sys.argv) < 2:
		t_num = 1
	else:
		t_num = sys.argv[1]
	print(t_num)
	for i in range(5):
		acc_i = ensemble(int(t_num))
		min_acc = min(acc_i, min_acc)
		max_acc = max(acc_i, max_acc)
		acc_ave += acc_i
	acc_ave /= 5
	print("average acc:", acc_ave, "max_acc:", max_acc, "min_acc", min_acc)
	pure_svm()
