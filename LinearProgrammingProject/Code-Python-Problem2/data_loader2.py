from scipy.io import loadmat
import numpy as np
import sympy
import scipy.optimize as optimize


def get_rid_of_dependent_rows(A,b):
    _, inds = sympy.Matrix(A).T.rref()
    all_row_indices = [i for i in range(0,A.shape[0])]
    dependent_row_indices = list(set(all_row_indices)-set(inds))
    A = np.delete(A,np.array(dependent_row_indices),axis=0)
    b = np.delete(b,np.array(dependent_row_indices),axis=0)
    return (A,b)

data = loadmat("Problem_2-7.mat")
A = data['A']
b = data['b']
epsilon = data['epsilon']
N = data['N']

A_T = np.transpose(A)
A_TA = np.dot(A_T,A)
A_Tb = np.dot(A_T,b)

A_arr = np.hstack((A_TA, -1*A_TA))
A_arr = np.vstack((A_arr, (-1 * A_arr)))
m_epsilon = N[0,0] * epsilon[0,0]
m_epsilon = m_epsilon * np.ones(A_arr.shape[0])

b_arr_1 = A_Tb + m_epsilon
b_arr_2 = m_epsilon - A_Tb

b_arr = np.vstack((b_arr_1,b_arr_2))
b_list = []
for i in range(0,b_arr.shape[0]):
    b_list.append(b_arr[i,0])
b_arr = np.array(b_list)
print(b_arr)

c = np.ones(36)

res = optimize.linprog(c,A_ub=A_arr,b_ub=b_arr,bounds=(0,None),method='simplex')
print(res.status)
print(res.fun)
print(res.nit)
print(res.x)
print("\n\n-----------------------------------------------------------------------------------------------\n\n")

np.save("P2_A_Gen",A_arr)
np.save("P2_b_Gen",b_arr)
np.save("P2_c_Gen",c)


''' ------- validate standard form -----'''

e = np.eye(A_TA.shape[0])
z = np.zeros((A_TA.shape[0],A_TA.shape[0]))

ez = np.hstack((e,z))
ze = np.hstack((z,e))

EZ = np.vstack((ez,ze))

A_arr = np.hstack((A_arr, EZ))

A_arr, b_arr = get_rid_of_dependent_rows(A_arr, b_arr)

print(A_arr.shape)
print(b_arr.shape)

c = np.array(np.ones(36).tolist() + np.zeros(36).tolist())

if np.linalg.matrix_rank(A_arr,tol=1e-20) != A_arr.shape[0]:
    print(" Rows of A are not INDEPENDENT \n")
    exit(0)

res = optimize.linprog(c,A_eq=A_arr,b_eq=b_arr,bounds=(0,None),method='simplex')
print(res.status)
print(res.fun)
print(res.nit)

'''------------ BIG M formulation----------------'''
I = np.identity(A_arr.shape[0])
A_arr = np.hstack((A_arr, I))
c = np.array(np.ones(36).tolist() + np.zeros(36).tolist() + [999.0 for i in range(0,A_arr.shape[0])])

res = optimize.linprog(c,A_eq=A_arr,b_eq=b_arr,bounds=(0,None),method='simplex')
print(res.status)
print(res.fun)
print(res.nit)

basic_solution_list = []
b_list = b_arr.tolist()

for i in range(0,72):
    basic_solution_list.append(0.0)
for b_val in b_list:
    basic_solution_list.append(b_val)

initial_basic_feasible_solution = np.array(basic_solution_list)

np.save("P2_A",A_arr)
np.save("P2_b",b_arr)
np.save("P2_c",c)
np.save("P2_bfs",initial_basic_feasible_solution)

