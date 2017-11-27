#!/usr/bin/env python

from sys import argv
from matplotlib import pyplot
from matplotlib import lines as mlines


def make_figure(x_vals, y_precision, y_recall, y_f1, x_label, y_label="Key Figures", save_as=False):
    """Draw a graph based on the values specified"""
    p_line = mlines.Line2D([], [], label="Precision", linestyle="--", color="red")
    r_line = mlines.Line2D([], [], label="Recall", linestyle="-.", color="green")
    f_line = mlines.Line2D([], [], label="F1", linestyle=None, color="blue")

    pyplot.plot(x_vals, y_precision, "r--")
    pyplot.plot(x_vals, y_recall, "g-.")
    pyplot.plot(x_vals, y_f1, "b")
    pyplot.legend(handles=[p_line, r_line, f_line])
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)

    if save and save_as is not None:
        pyplot.savefig(save_as)
    else:
        pyplot.show()

    pyplot.close()


save = False
knn = False
rf = False
mlp = False
read = False
file_name = "figure.png"

if len(argv) > 1:
    for arg in argv:
        arg = arg.lower()
        if arg in "save" and arg[0] == "s":
            save = True
        if arg in "read" and arg[0] == "r":
            read = True
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

if not knn and not rf and not mlp and not read:
    print("You have to specify either 'read', 'knn', 'rf' or 'mlp'!")
    exit(1)


if read:
    lines = []
    with open("c4-results.txt") as result_file:
        lines = result_file.readlines()

    results = {"knn": [], "forest": [], "neural": [], "bayes": []}
    section = None

    for line in [l.strip() for l in lines if len(l.strip()) > 0]:
        if section is None:
            section = line
        elif line.startswith("end "):
            section = None
        else:
            result = line.split()
            results[section].append(result)

    results.pop("bayes")

    for classifier in results:
        if classifier == "knn":
            file_name = "knn"
            x_label = "Number of Neighbours taken into account"
        elif classifier == "forest":
            file_name = "forests"
            x_label = "Number of Trees generated"
        elif classifier == "neural":
            file_name = "mlp"
            x_label = "Number of Hidden Layers (with 100 perceptrons each)"
        else:
            x_label = "Classifier Parameter"

        x_values = []
        y_pre = []
        y_rec = []
        y_f1 = []

        for (x, p, r, f) in results[classifier]:
            x_values.append(int(x))
            y_pre.append(float(p))
            y_rec.append(float(r))
            y_f1.append(float(f))

        file_name = "c4-{0}.png".format(file_name)
        make_figure(x_values, y_pre, y_rec, y_f1, x_label, "Key Figures", file_name)
else:
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

    make_figure(x, y_precision, y_recall, y_f1, x_label, "Key Figures", "figure.png")