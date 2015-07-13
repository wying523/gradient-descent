'''
  (c) 2015 Akshay Chiwhane
  Released under MIT License
'''
from __future__ import division # for floating point div
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import matplotlib.pyplot as plt


def hypothesis(t0, t1, x):
  return t0 + t1 * x


def gradient_descent(x, y, alpha, numiter):
  m = x.shape[0]
  t0 = [0]
  t1 = [0]

  cost = 0
  for i in range(1, m + 1):
    cost += (hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]) ** 2

  isconverged = False
  counter = 0

  while not isconverged and counter < numiter:

    t0_tmp_sum = 0
    t1_tmp_sum = 0
    for i in range(1, m + 1):
      t0_tmp_sum += hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]
      t1_tmp_sum += (hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]) * x[i - 1]


    t0_tmp = t0[len(t0) - 1] - alpha * t0_tmp_sum / m
    t1_tmp = t1[len(t1) - 1] - alpha * t1_tmp_sum / m

    t0.append(t0_tmp)
    t1.append(t1_tmp)

    cost_tmp = 0
    for i in range(1, m + 1):
      cost_tmp += (hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]) ** 2

    if abs(cost - cost_tmp) < 0.001:
      isconverged = True

    counter = counter + 1

  return t0, t1


def main():

  filename = 'data1.csv'

  data = list(csv.reader(open(filename, 'r'), delimiter = ','))

  x_vals = np.asarray([i[0] for i in data], dtype=np.float)
  y_vals = np.asarray([i[1] for i in data], dtype=np.float)
  

  #x_vals = np.loadtxt('ex2x.dat')
  #y_vals = np.loadtxt('ex2y.dat')

  # make sure to choose alpha wisely or else overflow errors can occur

  alpha = 0.01
  numiter = 2000

  t0, t1 = gradient_descent(x_vals, y_vals, alpha, numiter)

  intercept =  t0[len(t0) - 1]
  slope =  t1[len(t1) - 1]

  print intercept
  print slope

  plt.scatter(x_vals, y_vals, c='r', marker='x')
  plt.plot(x_vals, [intercept + slope * i for i in x_vals])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()








if __name__ == "__main__":
  main()
  
  