from sklearn import svm
from sklearn import tree
import numpy as np
import random

all_data = np.load('../course_train.npy')

def get_data():
	x = {}
	y = {}
	x_test = []
	y_test = []
	num = [0]*10
	for i in range(10):
		d = {}
		x[i] = {}
		dy = {}
		y[i] = {}
	# 加入正例
	for line in all_data:
		if line[-2] not in x[line[-1]]:
			if len(x[line[-1]]) >= 5:
				x_test.append(line[:-2])
				y_test.append(line[-1])
			else:
				a = []
				a.append(line[:-2])
				b = [1]
				x[line[-1]][line[-2]] = a
				y[line[-1]][line[-2]] = b
		else:
			x[line[-1]][line[-2]].append(line[:-2])
			y[line[-1]][line[-2]].append(1)
	# 加入负例
	for line in all_data:
		for i in range(10):
			if (i != line[-1]) and (line[-2] in x[i]):
				r = random.randint(0,9)
				if r==i:
					for j in x[i].keys():
						x[i][j].append(line[:-2])
						y[i][j].append(0)

	return x, y, x_test, y_test


def ensemble(x, y, x_test, y_test):
	prob = []
	for i in range(10):
		a = []
		for j in range(len(y_test)):
			a.append(0)
		prob.append(a)
	count = 0
	for i in range(10):
		for j in x[i].keys():
			#clf = tree.DecisionTreeClassifier()
			clf = svm.LinearSVC()
			clf = clf.fit(x[i][j], y[i][j])
			p = clf.predict(x_test)
			print("count:", count)
			count += 1
			for k in range(len(y_test)):
				prob[i][k] += p[k]

	acc = 0
	for i in range(len(y_test)):
		ml = 0
		m = 0
		for j in range(10):
			if int(prob[j][i]) > m:
				m = int(prob[j][i])
				ml = j
		if ml == y_test[i]:
			acc += 1
	print('ensemble acc:', acc/len(y_test))
	return acc/len(y_test)


if __name__ == '__main__':
	max_acc = 0
	min_acc = 1
	ave_acc = 0
	for i in range(5):
		x, y, x_test, y_test = get_data()
		acc = ensemble(x, y, x_test, y_test)
		max_acc = max(acc, max_acc)
		min_acc = min(acc, min_acc)
		ave_acc += acc
	print('max:', max_acc, "min:", min_acc, "average:", ave_acc/5)
	#pure_svr(x, y, x_test, y_test)

