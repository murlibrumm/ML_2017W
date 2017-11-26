#!/usr/bin/env python

from sys import argv
from matplotlib import pyplot

save = False
if len(argv) > 1:
    for arg in argv:
        arg = arg.lower()
        if arg in "save" and arg[0] == "s":
            save = True

knn = False
rf = True


# for knn
if knn:
    x = [5, 10, 25, 50, 100, 250]
    y_precision = [0.72, 0.75, 0.74, 0.7, 0.69, 0.64]
    y_recall = [0.74, 0.78, 0.76, 0.71, 0.7, 0.68]
    y_f1 = [0.73, 0.75, 0.71, 0.64, 0.62, 0.57]

if rf:
    x = [10, 20, 50, 250, 500, 1000]
    y_precision = [0.77, 0.78, 0.79, 0.79, 0.8, 0.79]
    y_recall = [0.79, 0.81, 0.81, 0.82, 0.82, 0.81]
    y_f1 = [0.77, 0.78, 0.79, 0.79, 0.79, 0.78]


pyplot.plot(x, y_precision, "r", x, y_recall, "g", x, y_f1, "b")

if save:
    pyplot.savefig("figure.png")
else:
    pyplot.show()

pyplot.close()