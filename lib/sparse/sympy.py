import numpy as np
from sympy import *
DEGREE = lambda b: max(degree(j) for a in b for j in (numer(a), denom(a)))

def inv(a, x):
    inv = x
    assert inv.cols == a.cols

    def SCALE(j, zz):
        for i in range(a.rows):
            a[i, j] = cancel(a[i, j] * zz)
        for i in range(inv.rows):
            inv[i, j] = cancel(inv[i, j] * zz)
        pass
        pass

    def ADD(i, j, zz):
        for l in range(a.rows):
            a[l, j] = cancel(a[l, j] + a[l, i] * zz)
        for l in range(inv.rows):
            inv[l, j] = cancel(inv[l, j] + inv[l, i] * zz)

    def SWAP(i, j):
        for l in range(a.rows):
            a[l, i], a[l, j] = a[l, j], a[l, i]
        for l in range(inv.rows):
            inv[l, i], inv[l, j] = inv[l, j], inv[l, i]

    for i in range(a.cols):
        ipivot = i
        deg = None
        best = None
        while ipivot < a.cols:
            if a[i, ipivot] != 0:
                d = degree(numer(a[i, ipivot]))
                if deg is None or d < deg:
                    best = ipivot
                    deg = d
            ipivot += 1
        if best is None:
            print(i)
            print(a[i, :])
            print(a)
            raise Exception("singular matrix")
        else:
            if best != i:
                SWAP(best, i)
        SCALE(i, 1/a[i, i])
        for j in range(i + 1, a.cols):
            ADD(i, j, -a[i, j])
        print("forward pass. column", i, "degree", DEGREE(a))
        pass
    for i in range(a.cols-1, -1, -1):
        pass
        pass
        print("backward pass. column", i, "degree", DEGREE(inv))
        for j in range(i):
            for l in range(inv.rows):
                inv[l, j] = cancel(inv[l, j] - inv[l, i] * a[i, j])
    #print(inv)
    #print(a)
    inv = inv
    return inv

def inverse(M, A, indices_from, indices_to):
    d, (r, c) = A
    m = SparseMatrix(M, M, {(r[j], c[j]): d[j] for j in range(len(d))})
    mm = inv(m, SparseMatrix(len((indices_from)), M, {(j, indices_from[j]): 1
        for j in range(len(indices_from))}))
    #mm = inv(m, SparseMatrix(M, len(indices_to), {(indices_to[j], j): 1
    #    for j in range(len(indices_to))}))
    t = zeros(len(indices_from), len(indices_to))
    for i in range(len(indices_from)):
        for j in range(len(indices_to)):
            t[i, j] = mm[i, indices_to[j]]
            #t[i, j] = mm[indices_from[i], j]
            #t[i, j] = mm[indices_from[i], indices_to[j]]
    return t

def steady(M, A):
    """
    If the left null space of A is one-dimensional and
    contains a vector with v[M - 1] != 0, this function
    finds one of those vectors.
    """
    d, (r, c) = A
    sparse_without_last_line = {(r[j], c[j]): d[j] for j in range(len(d))
                                                   if c[j] < M - 1}
    sparse_without_last_line[(M-1, M-1)] = 1
    m = SparseMatrix(M, M, sparse_without_last_line)
    mm = inv(m, SparseMatrix(1, M, {(0, M-1): 1}))
    return mm
