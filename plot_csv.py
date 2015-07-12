#generates a scatterplot with a given .csv file
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
#from optparse import OptionParser


def main():
  data = list(csv.reader(open(sys.argv[1], "r"), delimiter=","))
  data.pop(0)
  x = np.asarray([i[1] for i in data])
  y = np.asarray([i[2] for i in data])
  plt.scatter(x, y)
  plt.show()

  #TODO - add linear regression steps

if __name__ == "__main__":
  main()
  
  