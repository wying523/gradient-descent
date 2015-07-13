
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
#from optparse import OptionParser


def main():
  # data taken from 
  # https://github.com/vincentarelbundock/Rdatasets/blob/master/csv/datasets/AirPassengers.csv
  data = list(csv.reader(open("AirPassengers.csv", "r"), delimiter=","))

  #remove labels
  data.pop(0)

  #remove unnecessary 1st column
  for i in data:
    i.pop(0)

  x = np.asarray([i[0] for i in data])
  y = np.asarray([i[1] for i in data])
  plt.xlabel("Year")
  plt.ylabel('Number of Passengers (in thousands)')
  plt.title("Air Passengers")
  plt.scatter(x, y)
  plt.show()

if __name__ == "__main__":
  main()
  
  