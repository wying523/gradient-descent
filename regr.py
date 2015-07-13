'''
  (c) 2015 Akshay Chiwhane
  Released under MIT License

'''

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from __future__ import division # for floating point div


'''
  REQUIRES:   non-empty list x
  MODIFIES:   none
  EFFECTS:    regression model for a given x
'''
def h(x, theta_0, theta_1):

  return theta_0 + (theta_1 * x)



'''
  REQUIRES:   non-empty lists x and y
  MODIFIES:   theta_0, theta_1
  EFFECTS:    returns the slope and y-intercept of regression for one "step"

'''
def regression_step(x, y, alpha, m, theta_0, theta_1):
  sum_diff_0 = 0 # diff term for theta_0 calculation
  sum_diff_1 = 0 # diff term for theta_1 calculation

  for i in range(1, m + 1): # we want to sum from 1, 2, ..., m
    temp = h(x[i - 1], theta_0, theta_1) - y[i - 1]
    sum_diff_0 += temp
    sum_diff_1 += temp * x[i - 1]

  theta_0 = theta_0 - alpha * sum_diff_0 / m
  theta_1 = theta_1 - alpha * sum_diff_1 / m

  return theta_0, theta_1






def main():
  # data taken from 
  # https://github.com/vincentarelbundock/Rdatasets/blob/master/csv/datasets/AirPassengers.csv
  data = list(csv.reader(open("AirPassengers.csv", "r"), delimiter=","))

  #remove labels
  data.pop(0)

  #remove unnecessary 1st column
  for i in data:
    i.pop(0)

  # convert each elem to float val for consistency
  x = np.asarray([float(i[0]) for i in data])
  y = np.asarray([float(i[1]) for i in data])


  # placeholder until regression function is completed
  slope, intercept, _, _, _ = stats.linregress(x,y)

  linreg_y = [((slope * i) + intercept) for i in x]

  plt.xlabel("Year")
  plt.ylabel("Number of Passengers (in thousands)")
  plt.title("Air Passengers")
  plt.scatter(x, y)
  plt.plot(x, linreg_y, 'r')


  # need some way to animate plot at each step -- TODO
  plt.show()



if __name__ == "__main__":
  main()
  
  