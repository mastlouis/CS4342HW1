import numpy as np

def problem1(A, B):
    return A + B


def problem2(A, B, C):
    return A.dot(B) - C


def problem3(A, B, C):
    return (A * B) + C.T


def problem4(x, y):
    return x.dot(y)


def problem5(A):
    return np.zeros(A.shape)


def problem6(A):
    return np.ones(A.shape)


def problem7(A, alpha):
    return A + (np.eye(A.shape[0], A.shape[1]) * alpha)


def problem8(A, i, j):
    return A[i, j]


def problem9(A, i):
    return A([[i]]).sum()


def problem10(A, c, d):
    return A[np.logical_and(c <= A, A <= d)].mean()


def problem11(A, k):
    return np.linalg.eig(A)[1]


def problem12(A, x):
    return np.linalg.solve(A, x)


def problem13(A, x):
    return np.linalg.solve(A.T, x.T).T


if __name__ == "__main__":
    A = np.array([[1, 2, 3], [6, 5, 4], [11, 19, 23]])
    i = 1
    print(np.sum(A[i:i+1]))
