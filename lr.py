import numpy as np
from matplotlib import pyplot as plt


x_points = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y_points = np.array([1, 2, 3, 1, 4, 5, 6, 4, 7 , 10, 15, 9])

m = 0
c = 0

def calculate_y(x_points, m, c):
	y = m * x_points + c
	return y

def plot_line(y, data_points):
	plt.plot(data_points, y, 'r')

#plot_line(y, x_points)
#plt.show()
def calculate_loss(y, y_points):
	return np.linalg.norm(y - y_points)

def grad_descent(x_points, y_points, y):
	dm = np.sum((y - y_points) * x_points)
	dc = np.sum((y - y_points))
	return dm, dc

learning_rate = 0.001
print(m, c)
for i in range(10):
	y = calculate_y(x_points, m, c)
	loss = calculate_loss(y, y_points)
	lossText = "loss: "+str(loss)+"\n"+"m: "+str(m)+"\n"+"c: "+str(c)
	plt.text(1, 13, lossText)
	plt.plot(x_points, y_points, 'ro')
	plt.plot(x_points, y)
	plt.show()
	dm, dc = grad_descent(x_points, y_points, y)
	m = m - learning_rate * dm
	c = c - learning_rate * dc

