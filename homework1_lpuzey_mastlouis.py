# Linda Puzey
# Machine Learning Homework 1

import numpy as np


# Part 1

# Q1: Given matrices A and B, compute and return an expression for A +B
def problem1(A, B):
    return A + B


# Q2: Given matrices A,B, and C compute and return AB - C
# (Right-multiply martric A by matrix B and then subtract C
# Use dot or np.dot
def problem2(A, B, C):
    return np.dot(A, B) - C


# Q3: Given matrices A,B, and C, return A dot B + C (transpose)
# dot is * in numpy
def problem3(A, B, C):
    return (A*B) + C.T


# Q4: Given x and y, compute the inner product of x and y
#xTy
def problem4(x, y):
    return np.inner(x,y)


# Q5: Given a matrix A, return a vector with the same number of rows as A but that contains all zeros
# Use np.zeros
def problem5(A):
    d = np.shape(A)
    return np.zeros(d[0])


# Q6: Given a matrix A, return a vector with the same number of rows as A but that all contains ones
# use np.ones
def problem6(A):
    d = np.shape(A)
    return np.ones(d[0])


# Q7: Given a square matrix A and scalar alpha
# Compute A + alpha I
# I is the identity matrix with the same dimentions as A
# Use np.eye
def problem7(A, alpha):
    return A + alpha * np.eye(A.shape)


# Q8: Given matrix A and integers i, j
# return the jth column of the ith row of A
def problem8(A, i, j):
    return A[i][j]


# Q9: Given matrix A and integer i
# return the sum of all the entries in the ith row
# Don't use a loop
# Use np.sum
def problem9(A, i):
    sum = np.sum(A[i])
    return sum


# Q10: Given matrix A and scalars c,d
# Compute the arithmetic mean over all entries of A that are between c and d
# then compute 1/|S|sum of A
# use np.nonzero and np.mean
def problem10(A, c, d):
    g = A[np.nonzero(A > c)]
    l = g[np.nonzero(g < d)]
    return np.mean(l)


# Q11: Given an n x n matrix A and an integer k
# return an n x k matrix containg the right eigenvectors of A corresponding to the k largest eigenvalues of A
# Use np.linalg.eig
def problem11(A, k):
    eig = np.linalg.eig(A)[1]
    c = A.shape - k
    return eig[:,c:]


# Q12: Given square matrix A and column vector x
# compute A^-1x
# Use np.linalg.solve
# Don't use linalg.inv or ** -1
def problem12(A, x):
    return np.linalg.solve(A,x)


# Q13: Given square matrix A and row vector x
# compute xA^-1
# Hint AB = (BT AT)T
# Use np.linalg.solve
def problem13(A, x):
    y = np.linalg.solve(A.T,x.T)
    return y.T


# Tests
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
z = np.array([[9, 10], [11, 12]])
a = np.array([[1, 2], [1, 2], [1, 2]])

print("Problem1:\n", problem1(x, y))
print("Problem2:\n", problem2(x, y, z))
print("Problem3:\n", problem3(x, y, z))
print("Problem4:\n", problem4(x, y))
print("Problem5:\n", problem5(a))
print("Problem6:\n", problem6(a))
print("Problem7:\n", problem7(a,2))
print("Problem8:\n", problem8(y,1,1))
print("Problem9:\n", problem9(x,1))

print("Problem11:\n", problem11(x,1))
