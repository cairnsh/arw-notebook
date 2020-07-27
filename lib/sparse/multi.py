import numpy as np, scipy as sp
import time
import scipy.linalg, scipy.sparse.linalg
from mpmath import mp

def _TO_NUMPY(A):
    return np.array(A.tolist(), dtype=float)

def FROM_NUMPY(foo):
    try:
        return mp.matrix(foo)
    except TypeError:
        pass
    rows, cols = foo.shape
    m = mp.matrix(rows, cols)
    col = 0
    for j in range(len(foo.data)):
        while foo.indptr[col + 1] <= j:
            col += 1
        m[foo.indices[j], col] = foo.data[j]
    return m

def sparse_transposition(A):
    d, (r, c) = A
    return d, (c, r)

def sparse_make_single_precision(M, A):
    d, (r, c) = A
    af = [float(z) for z in d], (r, c)
    return sp.sparse.csc_matrix(af, (M, M))

def sparse_single_precision_solve(a, b):
    b = _TO_NUMPY(b)
    x = sp.sparse.linalg.spsolve(a, b, use_umfpack=False)
    x = FROM_NUMPY(x) #
    return x

def mmul(A, x):
    d, (r, c) = A
    z = mp.matrix(x.rows, x.cols)
    for j in range(len(d)):
        for i in range(x.cols):
            if x[c[j], i] != 0:
                z[r[j],i]+=d[j]*x[c[j],i]
    return z

def solve_multiprecision(M, A, b, tolerance=1e-60):
    af = sparse_make_single_precision(M, A)
    so = sp.sparse.linalg.factorized(af)
    def solve(b):
        b = _TO_NUMPY(b).flatten()
        x = so(b)
        return FROM_NUMPY(x)
    """
    def solve(b):
        b = _TO_NUMPY(b)
        x, e = sp.sparse.linalg.gmres(af, b)
        if e != 0:
            print("error", e)
            raise Exception("gmres error")
        return FROM_NUMPY(x)
    """
    x = solve(b)
    current_prec = mp.prec
    try:
        guess = np.log2(M) - np.log2(tolerance) + 24
        mp.prec = guess
        while True:
            residual = b - mmul(A, x)
            scale = mp.norm(residual, p='inf')
            print("residual: %e" % scale)
            if scale < tolerance:
                return x
            compare = solve(residual / scale)
            x += scale * compare
            amplification = float(mp.norm(compare, p='inf'))
            mp.prec = max(mp.prec, guess + np.log2(amplification))
            print("precision:", mp.prec)
    finally:
        mp.prec = current_prec

def inverse(M, A, indices_from, indices_to):
    t = time.time()
    b = mp.matrix(M, len(indices_to))
    for j in range(len(indices_to)):
        b[indices_to[j], j] = 1
    x = solve_multiprecision(M, A, b)
    answer = mp.matrix(len(indices_from), len(indices_to))
    for i in range(len(indices_from)):
        for j in range(len(indices_to)):
            answer[i, j] = x[indices_from[i], j]
    print("inverse runtime was %f" % (time.time() - t))
    return answer
