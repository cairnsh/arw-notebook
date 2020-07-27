import numpy as np, scipy as sp
from flint import fmpq_mat, fmpz, fmpq
import time
import scipy.linalg

def cfrac(q):
    out = []
    minus = 1
    if q < 0:
        minus = -1
        q = -q

    if q < 1:
        out.append(0)
        q = 1/q
    for j in range(10):
        z = np.floor(q)
        z = int(z)
        out.append(z)
        q -= z
        if q < 1e-7:
            break
        else:
            q = 1/q
    return minus, out

def guessrational(q):
    if isinstance(q, fmpq):
        return q
    minus, fraction = cfrac(q)
    numer = 1
    denom = 0
    for j in fraction[::-1]:
        denom, numer = numer, denom
        numer += denom * j
    numer*=minus
    return fmpq(numer, denom)

def inverse(M, A, indices_from, indices_to):
    m = fmpq_mat(M, M)
    d, (r, c) = A
    for j in range(len(d)):
        a = guessrational(d[j])
        #print("guessed", d[j], "=", a)
        m[r[j], c[j]] = guessrational(d[j])
    t = fmpq_mat(len(indices_from), len(indices_to))
    for j in range(len(indices_to)):
        z = fmpq_mat(M, 1)
        z[indices_to[j], 0] = 1
        o = m.solve(z)
        for i in range(len(indices_from)):
            t[i, j] = o[indices_from[i], 0]
    return t
