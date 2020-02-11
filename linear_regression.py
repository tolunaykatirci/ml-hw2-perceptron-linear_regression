import numpy as np
from random import uniform, randint


def average(list):
    return 1.0 * sum(list) / len(list)


def generate_data(N_Samples, N_Points):
    total = N_Samples + N_Points

    # generate target function (represented by vector w)
    x1 = uniform(-1, 1)  # random value in range [-1,1]
    y1 = uniform(-1, 1)
    x2 = uniform(-1, 1)
    y2 = uniform(-1, 1)

    w = [1, 0, 0]
    w[1] = (x2 - x1) / (y1 - y2)
    w[2] = - (x1 * w[0] + y1 * w[1])
    w = np.array([[w[0]], [w[1]], [w[2]]])

    # generate n random points between [-1,1]X[-1,1]
    X = np.random.rand(total, 3) * 2 - 1
    X[:, 0] = 1


    Y = np.dot(X, w)
    Y[Y >= 0] = 1
    Y[Y < 0] = -1
    Y = np.int16(Y)

    X_train = X[0:N_Points, :]
    X_test = X[N_Points:total, :]
    Y_train = Y[0:N_Points, :]
    Y_test = Y[N_Points:total, :]

    return w, X_train, Y_train, X_test, Y_test


def learn_w(X, Y):
    X_pinv = np.linalg.pinv(X)
    w = np.dot(X_pinv, Y)
    return w


def evaluate_e_in(X, Y, w, N):
    guess = np.dot(X, w)
    guess[guess >= 0] = 1
    guess[guess < 0] = -1
    guess = np.int16(guess)
    correct = sum(guess == Y)
    return 1.0 - correct * 1.0 / N


def evaluate_e_out(X_test, Y_test, w , N_test):
    guess = np.dot(X_test, w)
    guess[guess >= 0] = 1
    guess[guess < 0] = -1
    guess = np.int16(guess)
    correct = sum(guess == Y_test)
    return 1.0 - correct * 1.0 / N_test


def linear_regression(N_Samples, N_Points = 10):

    E_in_list = []
    E_out_list = []

    for i in range(1000):
        w, X_train, Y_train, X_test, Y_test = generate_data(N_Samples, N_Points)
        w = learn_w(X_train, Y_train)

        E_in_list.append(evaluate_e_in(X_train, Y_train, w, X_train.shape[0]))
        E_out_list.append(evaluate_e_out(X_test, Y_test, w, X_test.shape[0]))

    print("The average E_in: ", average(E_in_list))
    print("The average E_out: ", average(E_out_list))


def generate_nonlinear_data(n):
    X = np.random.rand(n, 3) * 2 - 1
    X[:, 0] = 1

    # f(x1, x2) = sign(x21 + x22 − 0.6)
    Y = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        Y[i, 0] = X[i, 1] * X[i, 1] + X[i, 2] * X[i, 2] - 0.6

    Y[Y >= 0] = 1
    Y[Y < 0] = -1
    Y = np.int16(Y)
    return X, Y

def nonlinear_transformation(N_Samples, N_Points = 1000):

    E_in_list = []
    E_out_list = []

    for i in range(1000):
        X, Y = generate_nonlinear_data(N_Samples + N_Points)
        X_trans = np.zeros((X.shape[0], 6))
        for j in range(Y.shape[0]):
            X_trans[j, 0:3] = X[j, :]
            X_trans[j, 3] = X[j, 1] * X[j, 2]
            X_trans[j, 4] = X[j, 1] * X[j, 1]
            X_trans[j, 5] = X[j, 2] * X[j, 2]
            if randint(1, 10) == 1:
                # 10% to be wrong on original data
                Y[j, 0] = -Y[j, 0]

        X_train = X_trans[:N_Points, :]
        X_test = X_trans[N_Points:, :]
        Y_train = Y[:N_Points, :]
        Y_test = Y[N_Points:, :]

        w = learn_w(X_train, Y_train)

        E_in_list.append(evaluate_e_in(X_train, Y_train, w, X_train.shape[0]))
        E_out_list.append(evaluate_e_out(X_test, Y_test, w, X_test.shape[0]))

    print(average(E_in_list))
    print(average(E_out_list))


