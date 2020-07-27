import numpy as np, scipy as sp
import time
import scipy.linalg, scipy.sparse.linalg

import subprocess, asyncio, os

class DoneException(Exception):
    pass

INVERSE = "/home/gibson/sdb2/cola/ass/bar/sparse/inverse_more_memory"

def maintain(lines):
    roc = subprocess.Popen(
        ["/usr/bin/mpirun", "-np", "4", INVERSE],
        bufsize=0,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    os.set_blocking(roc.stdout.fileno(), 0)
    outs = []
    def read():
        zz = roc.stdout.read()
        if zz is None:
            pass
        elif len(zz) == 0:
            raise DoneException()
        else:
            outs.append(zz)
    try:
        for line in lines:
            line = str(line) + "\n"
            line = line.encode()
            upto = 0
            while upto < len(line):
                inc = roc.stdin.write(line[upto:])
                read()
                if inc is None:
                    time.sleep(0.1)
                else:
                    upto += inc
        while True:
            read()
            time.sleep(0.1)
    except DoneException:
        return b"".join(outs)

def inverse(N, data, indices_from, indices_to):
    """
        compute inv(M)[indices_from, indices_to] using
        the sparse linear algebra program called MUMPS.
        N is the size of the matrix.
        data = d, (r, c) is the sparse data.
    """

    "build the input to the program"
    d, (r, c) = data
    M = len(d)
    lines = [N, M]
    for i in range(M):
        lines.append("%s,%s,%s" % (1+r[i], 1+c[i], d[i]))
    ifp = np.argsort(indices_from)
    itp = np.argsort(indices_to)
    indices_from = np.take(indices_from, ifp)
    indices_to = np.take(indices_to, itp)
    mf = len(indices_from)
    mt = len(indices_to)
    lines.append(mf * mt)
    for j in range(mt):
        for i in range(mf):
            lines.append(1 + indices_from[i])
    lines.append(N)
    ix = 0
    offset = 0
    for j in range(N):
        lines.append(1 + offset)
        if ix < len(indices_to) and indices_to[ix] == j:
            ix += 1
            offset += mf
    assert ix == mt

    "run the program and extract the output"
    roc = maintain(lines).decode()
    where = roc.find("result\n")
    out = []
    if where >= 0:
        for line in roc[where:].split("\n")[1:]:
            if line.startswith("  "):
                out.append(float(line))
            else:
                break
    out = np.array(out).reshape(mt, mf).T
    out[ifp, :] = out
    out[:, itp] = out
    return out
