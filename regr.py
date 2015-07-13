'''
  (c) 2015 Akshay Chiwhane
  Released under MIT License
'''
from __future__ import division # for floating point div
from scipy import stats
import csv
import numpy as np
import matplotlib.pyplot as plt


def hypothesis(t0, t1, x):
  return t0 + t1 * x

def compute_cost(t0, t1, x, y, m):
  cost = 0
  for i in range(1, m + 1):
    cost += (hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]) ** 2
  return cost   


def compute_sums(t0, t1, x, y, m):  
  t0_tmp_sum = 0
  t1_tmp_sum = 0

  for i in range(1, m + 1):
    t0_tmp_sum += hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]
    t1_tmp_sum += (hypothesis(t0[len(t0) - 1], t1[len(t1) - 1], x[i - 1]) - y[i - 1]) * x[i - 1]

  return t0_tmp_sum, t1_tmp_sum



def gradient_descent(x, y, alpha, numiter):
  m = x.shape[0]

  t0 = [0]
  t1 = [0]

  cost = compute_cost(t0, t1, x, y, m)

  is_converged = False
  counter = 0

  while (not is_converged) and counter < numiter:
    t0_tmp_sum, t1_tmp_sum = compute_sums(t0, t1, x, y, m)

    t0_tmp = t0[len(t0) - 1] - alpha * t0_tmp_sum / m
    t1_tmp = t1[len(t1) - 1] - alpha * t1_tmp_sum / m

    t0.append(t0_tmp)
    t1.append(t1_tmp)

    cost_tmp = compute_cost(t0, t1, x, y, m)
    if abs(cost - cost_tmp) < 0.001:
      is_converged = True

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
  numiter = 5000

  t0, t1 = gradient_descent(x_vals, y_vals, alpha, numiter)
  print len(t0)
  print len(t1)

  intercept =  t0[len(t0) - 1]
  slope =  t1[len(t1) - 1]

  slope_act, intercept_act, r_value, p_value, std_err = stats.linregress(x_vals,y_vals)

  print 'our slope {} vs scipy stats{}'.format(slope, slope_act)
  print 'out intercept {} vs scipy stats {}'.format(intercept, intercept_act)


  plt.scatter(x_vals, y_vals, c='r', marker='x')

  plt.plot(x_vals, [intercept + slope * i for i in x_vals], c='g', label='our regression routine')
  plt.plot(x_vals, [intercept_act + slope_act * i for i in x_vals], c='b', label='scipy stats regression routine')


  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()



if __name__ == "__main__":
  main()
  
  