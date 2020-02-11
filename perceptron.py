# CSE4088 Homework #2
# Tolunay Katirci - 150115014

from random import uniform
from random import randint
import matplotlib,sys
import numpy as np
from matplotlib import pyplot as plt


# creates a random data set with default N = 100
def randomdata(N = 100):
    d = []
    for i in range(N):
        x = uniform(-1,1)   # random value in range [-1,1]
        y = uniform(-1,1)
        d.append([x, y])    # add to array
    return d


# computes a random line and returns a and b params: y = mx + n
def randomline():
    x1 = uniform(-1, 1)     # random value in range [-1,1]
    y1 = uniform(-1, 1)
    x2 = uniform(-1, 1)
    y2 = uniform(-1, 1)

    m = abs(y1 - y2) / abs(x1 - x2)
    n = y1 - m * x1
    return [m, n]   # mx + n


# maps a point (x1, y1) to a sign -1, +1 for the function f
def map_point(point, f):
    x1 = point[0]
    y1 = point[1]

    y = f(x1)
    compare_to = y1
    return sign(y, compare_to)


# returns +1 or -1 by comparing x to compare_to parameter (default = 0)
def sign(x,compare_to = 0):
    if x > compare_to: return +1
    else: return -1


# creates a misclassified set for each element of t_set
def create_misclassified_set(t_set, w):
    res = tuple()

    for i in range(len(t_set)):
        point = t_set[i][0]
        s = hyp(w, point)
        yn = t_set[i][1]
        if s != yn:
            res = res + (i,)
    return res


# Hypothesis function returns w0x0 + w1x1 + w2x2
def hyp(w, x):
    res = 0
    for i in range(len(x)):
        res = res + w[i]*x[i]
    return sign(res)


# returns
# t_set: item of t_set is: [[vector_x], y]
# w: vector of same dimension as vector_x of weights
# iteration: Number of iterations needed for convergence
# f: target lambda function f
def PLA(N_points = 100):
    N = N_points
    iteration = 0
    # create random data
    d = randomdata(N)
    # create random function
    l = randomline()
    f = lambda x: l[0] * x + l[1]
    # weight vector w0 , w1, w2
    w = [0, 0, 0]
    # build training set
    training_set = []

    for i in range(len(d)):
        x = d[i]
        y = map_point(x, f)  # map x to +1 or -1 for training points
        training_set.append([[1, x[0], x[1]], y])

    # iterate Perceptron Algorithm
    iterate = True
    while iterate:
        iteration = iteration + 1
        # pick a misclassified point from misclassified set
        misclassified_set = create_misclassified_set(training_set, w)
        # if there are no misclassified points break iteration weight are ok.
        if len(misclassified_set) == 0:
            # plot(training_set, l)
            # print(w)
            break
        # random misclassified index
        index = randint(0, len(misclassified_set) - 1)
        p = misclassified_set[index]
        point = training_set[p][0]

        s = hyp(w, point)
        yn = training_set[p][1]

        # update weights if misclassified
        if s != yn:
            xn = point
            w[0] = w[0] + yn * xn[0]
            w[1] = w[1] + yn * xn[1]
            w[2] = w[2] + yn * xn[2]
    return training_set, w, iteration, f


# Returns the average of difference between f and g (g is equivalent as vector w )
def calculate_diff(f, w):
    count = 0
    limit = 100
    diff = 0
    while count < limit:
        count = count + 1
        # generate random point as out of sample data
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        vector = [1, x, y]

        sign_f = sign(f(x), y)
        sign_g = hyp(w, vector)
        # check result and count if difference between target function f and hypothesis function g
        if sign_f != sign_g: diff = diff + 1

    return diff / (count * 1.0)


def run_PLA(N_samples, N_points):
    samples = []  # vector of 1 classified, 0 misclassified
    iterations = []  # vector of iterations needed for each PLA
    b_misclassified = False
    diff = []  # vector of difference average between f and g

    for i in range(N_samples):
        # run PLA in sample
        training_set, w, iteration, f = PLA(N_points)
        iterations.append(iteration)
        # check if points are classified or not
        for i in range(len(training_set)):
            point = training_set[i][0]
            s = hyp(w, point)
            yn = training_set[i][1]
            if yn != s:
                samples.append(0)
                b_misclassified = True
                break

        # check difference between f and g
        diff.append(calculate_diff(f, w))
        if not b_misclassified: samples.append(1)

        b_misclassified = False

    print('number of misclassified samples: %s ' % samples.count(0))
    print('number of classified samples: %s ' % samples.count(1))
    print('number of iteration average: %s ' % (str(sum(iterations) / len(iterations) * 1.0)))
    print("average of difference in function g: %s" % (sum(diff) / (len(diff) * 1.0)))


def plot(t_set, l):
    x = np.linspace(-1, 1, 100)
    label = 'y=%sx+%s' % (l[0], l[1])
    plt.plot(x, x*l[0]+l[1], label=label)
    for i in range(len(t_set)):
        point = t_set[i][0]
        color = "b"
        if t_set[i][1] == 1:
            color = "r"

        plt.scatter(point[1], point[2], c=color)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.show()


run_PLA(1000, 10)
