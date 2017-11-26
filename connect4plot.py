#!/usr/bin/env python

from sys import argv
from matplotlib import pyplot, lines

save = False
knn = False
rf = False
mlp = False
file_name = "figure.png"

if len(argv) > 1:
    for arg in argv:
        arg = arg.lower()
        if arg in "save" and arg[0] == "s":
            save = True
        if arg in "knn" and arg[0] == "k":
            knn = True
            rf = False
            mlp = False
            file_name = "c4-knn.png"
        if arg in "rf" and arg[0] == "r":
            knn = False
            rf = True
            mlp = False
            file_name = "c4-forests.png"
        if arg in "mlp" and arg[0] == "m":
            knn = False
            rf = False
            mlp = True
            file_name = "c4-mlp.png"

if not knn and not rf and not mlp:
    print("You have to specify either 'knn', 'rf' or 'mlp'!")
    exit(1)


# for knn
if knn:
    x_label = "Number of Neighbours taken into account"
    x = [5, 10, 25, 50, 100, 250]
    y_precision = [0.72, 0.75, 0.74, 0.7, 0.69, 0.64]
    y_recall = [0.74, 0.78, 0.76, 0.71, 0.7, 0.68]
    y_f1 = [0.73, 0.75, 0.71, 0.64, 0.62, 0.57]

if rf:
    x_label = "Number of Trees generated"
    x = [10, 20, 50, 250, 500, 1000]
    y_precision = [0.77, 0.78, 0.79, 0.79, 0.8, 0.79]
    y_recall = [0.79, 0.81, 0.81, 0.82, 0.82, 0.81]
    y_f1 = [0.77, 0.78, 0.79, 0.79, 0.79, 0.78]

if mlp:
    x_label = "Number of Hidden Layers (with 100 perceptrons each)"
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_precision = [0.78, 0.8, 0.8, 0.8, 0.79, 0.79, 0.79, 0.79, 0.79, 0.8]
    y_recall = [0.81, 0.82, 0.8, 0.8, 0.8, 0.79, 0.79, 0.8, 0.8, 0.8]
    y_f1 = [0.79, 0.81, 0.8, 0.8, 0.79, 0.79, 0.79, 0.79, 0.79, 0.8]

p_line = lines.Line2D([], [], label="Precision", linestyle="--", color="red")
r_line = lines.Line2D([], [], label="Recall", linestyle="-.", color="green")
f_line = lines.Line2D([], [], label="F1", linestyle=None, color="blue")

pyplot.plot(x, y_precision, "r--", x, y_recall, "g-.", x, y_f1, "b")
pyplot.legend(handles=[p_line, r_line, f_line])
pyplot.ylabel("")
pyplot.xlabel(x_label)

if save:
    pyplot.savefig(file_name)
else:
    pyplot.show()

pyplot.close()
