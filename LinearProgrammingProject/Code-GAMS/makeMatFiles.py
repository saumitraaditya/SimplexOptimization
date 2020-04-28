import scipy.io as io;
import numpy as np;

A = np.load("P1_A.npy")
b = np.load("P1_b.npy")
c = np.load("P1_c.npy")
data = {
    "A": A,
    "b": b,
    "c": c
}
io.savemat("P1.mat", data)

A = np.load("P2_A.npy")
b = np.load("P2_b.npy")
c = np.load("P2_c.npy")
data = {
    "A": A,
    "b": b,
    "c": c
}
io.savemat("P2.mat", data)

A = np.load("P2_A_Gen.npy")
b = np.load("P2_b_Gen.npy")
c = np.load("P2_c_Gen.npy")
data = {
    "A": A,
    "b": b,
    "c": c
}
io.savemat("P2_Gen.mat", data)