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

''' Read the raw data file'''
data = loadmat('Problem_1-3.mat')

''' A from data is a row vector each of whosel element is a 2D array with 162 columns and rows ranging from 15-20'''
A = data['A']

''' w is again a row vector with each element being a 2 D array but containing only one element'''
w = data['w']

''' d_pr is again a row vector with each element being a 2 D array but containing only one element'''
d_pr = data['d_pr']

''' U is a 2D array with just one element'''
U = data['U']

''' I total we have 162 'x' variables, 262 theta+ , 262 theta- and 262 slack variables'''

''' next step is to construct A, b, c from the raw data.'''

''' need a dummy row for using the function'''


'''-----------------theta_postives ------------------'''
theta = -999 * np.ones((1,262))
for i in range(0,A.shape[1]):
    coeffs = -999 * np.ones((A[0, i].shape[0], 1))
    for j in range(0,A.shape[1]):
        if i == j:
            # add ones and negatives
            # coeffs = np.hstack((coeffs,np.eye((A[0,i].shape[0], A[0,j].shape[0]))))
            coeffs = np.hstack((coeffs,np.eye(A[0,i].shape[0])))

        else:
            # add zeros
            coeffs = np.hstack((coeffs, np.zeros((A[0, i].shape[0], A[0, j].shape[0]))))
    # remove the first column
    coeffs = np.delete(coeffs, (0), axis=1)
    print(coeffs.shape)
    print(theta.shape)
    theta = np.vstack((theta, coeffs))

theta = np.delete(theta, (0), axis=0)
print(coeffs.shape)
print(theta.shape)

print(theta)

theta_positive = theta


'''-----------------theta_negatives ------------------'''

theta = -999 * np.ones((1,262))
for i in range(0,A.shape[1]):
    coeffs = -999 * np.ones((A[0, i].shape[0], 1))
    for j in range(0,A.shape[1]):
        if i == j:
            # add ones and negatives
            # coeffs = np.hstack((coeffs, -1 * np.eye((A[0,i].shape[0], A[0,j].shape[0]))))
            coeffs = np.hstack((coeffs, -1 * np.eye(A[0,i].shape[0])))

        else:
            # add zeros
            coeffs = np.hstack((coeffs, np.zeros((A[0, i].shape[0], A[0, j].shape[0]))))
    # remove the first column
    coeffs = np.delete(coeffs, (0), axis=1)
    print(coeffs.shape)
    print(theta.shape)
    theta = np.vstack((theta, coeffs))

theta = np.delete(theta, (0), axis=0)
print(coeffs.shape)
print(theta.shape)

print(theta)

theta_negative = theta


theta = np.hstack((theta_positive, theta_negative))

print(theta.shape)

'''------------------------------------ join A matrix --------------------------------------------'''

A_array = np.ones((1,162))
for i in range(0,A.shape[1]):
    A_mat = A[0,i].toarray()
    for j in range(A_mat.shape[0]):
        A_array = np.vstack((A_array, A_mat[j,:]))

A_array = np.delete(A_array, (0), axis=0)
print(A_array.shape)

''' --------------- join A and theta ---------------------'''
A_array = np.hstack((A_array, theta))
print(A_array.shape)

Aeq = A_array

''' ------- have coeffs for first set of constraints above -------------------------------------------'''

A = np.zeros((262,162))
theta = -1 * theta

Aub = np.hstack((A, theta))

print("A_array_2 shape is \n{}".format(Aub.shape))


''' ------- have coeffs for second set of constraints above -------------------------------------------'''


A = data['A']

d_pr_list = []
for i in range(0,d_pr.shape[1]):
    d_pr_list.append(float(d_pr[0,i][0,0]))
print(d_pr_list)


b_list = []
for i in range(0,A.shape[1]):
    for j in range(0,A[0,i].shape[0]):
        b_list.append(d_pr_list[i])

for i in range(0,A.shape[1]):
    for j in range(0,A[0,i].shape[0]):
        b_list.append(U[0,0] -d_pr_list[i])

print(len(b_list))
b_array = np.array(b_list)


beq = np.array(b_list[0:262])

bub = np.array(b_list[262:])

c_list = []
for i in range(0,162):
    c_list.append(0.0)
for i in range(0,A.shape[1]):
    for j in range(0,A[0,i].shape[0]):
        c_list.append(w[0,i][0,0])
for i in range(0,A.shape[1]):
    for j in range(0,A[0,i].shape[0]):
        c_list.append(w[0,i][0,0])
c_array = np.array(c_list)



res = optimize.linprog(c_array,A_ub=Aub,b_ub=bub,A_eq =Aeq, b_eq=beq, bounds=(0,None),method='interior-point')
print(res.status)
print(res.fun)
print(res.nit)


print(Aub.shape)
print(Aeq.shape)

''' prepare std form formulation'''
if np.linalg.matrix_rank(Aeq)!=Aeq.shape[0]:
    Aeq, beq = get_rid_of_dependent_rows(Aeq,beq)

if np.linalg.matrix_rank(Aub)!=Aub.shape[0]:
    Aub, bub = get_rid_of_dependent_rows(Aub,bub)

A1 = np.hstack((Aeq, np.zeros((Aub.shape[0],Aub.shape[0]))))
A2 = np.hstack((Aub, np.eye(Aub.shape[0])))

Aeq = np.vstack((A1,A2))
print(Aeq.shape)

''' prepare c vector by adding slack variables'''
for i in range(0,Aub.shape[0]):
    c_list.append(0)

''' combine b_array'''
b = np.array(b_list)


''' verify that std form formulation and general form has the same cost'''
c_array = np.array(c_list)
res = optimize.linprog(c_array,A_eq =Aeq, b_eq=b, bounds=(0,None),method='interior-point')
print(res.status)
print(res.fun)
print(res.nit)

'''prepare Big M formulation'''


b_list = b.tolist()

LARGE_VALUE = 9.0

''' should create automated approach for Big-M formulation.'''
num_dummy_variables = Aeq.shape[0]
''' extend cost vector'''
for i in range(0,num_dummy_variables):
    c_list.append(LARGE_VALUE)

c = np.array(c_list)

''' extend A array '''
identity = np.identity(num_dummy_variables)
Aeq = np.hstack((Aeq,identity))

''' initial basic solution'''
basic_solution_list = []
for i in range(0,Aeq.shape[1]-Aeq.shape[0]):
    basic_solution_list.append(0.0)
for b_val in b_list:
    basic_solution_list.append(b_val)



initial_basic_feasible_solution = np.array(basic_solution_list)

''' lets print summary of all input'''
print("\n-------------summary of all inputs to first problems including Big-M formulation is as below-------")
print("\nshape of coefficient matrix is {}".format(Aeq.shape))
print("\nshape of requirement matrix is {}".format(b.shape))
print("\nshape of cost matrix is {}".format(c.shape))
print("\nshape of initial bfs matrix is {}".format(initial_basic_feasible_solution.shape))

np.save("P1_A",Aeq)
np.save("P1_b",b)
np.save("P1_c",c)
np.save("P1_bfs",initial_basic_feasible_solution)

Aeq = Aeq.astype(np.longdouble)
b = b.astype(np.longdouble)
c = c.astype(np.longdouble)
initial_basic_feasible_solution = initial_basic_feasible_solution.astype(np.longdouble)

np.save("P1_A_ld",Aeq)
np.save("P1_b_ld",b)
np.save("P1_c_ld",c)
np.save("P1_bfs_ld",initial_basic_feasible_solution)

res = optimize.linprog(c,A_eq = Aeq, b_eq = b, bounds=(0,None),method='interior-point')
print(res.status)
print(res.fun)
print(res.nit)
























