#generates a scatterplot with a given .csv file
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
#from optparse import OptionParser


def main():
  data = list(csv.reader(open(sys.argv[1], "r"), delimiter=","))

  x = np.asarray([i[0] for i in data])
  y = np.asarray([i[1] for i in data])
  z = np.asarray([i[1] for i in data])
  plt.scatter(x, y, z)
  plt.show()

if __name__ == "__main__":
  main()
  
  