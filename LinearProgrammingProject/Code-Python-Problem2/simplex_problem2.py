
import numpy as np
from numpy.linalg import inv
import random
import scipy.io

''' indentifying linearly independent rows in a matrix
https://stackoverflow.com/questions/28816627/how-to-find-linearly-independent-rows-from-a-matrix
'''



def check_termination_condition(full_tableau):
    reduced_costs = full_tableau[0, 1:]
    # print("reduced costs row is \n{}".format(reduced_costs))
    flag = True
    for val in np.nditer(reduced_costs):
        if val<0.0:
            flag = False
            break
    return flag


def lexicographically_smaller(vec1, vec2):
    diff_vec = vec1 - vec2
    diff_vec = np.around(diff_vec,decimals=6)
    for val in np.nditer(diff_vec):
        if val !=0.0:
            if val < 0.0:
                return 1
            else:
                return 2
    return 0

''' return index of lexicographically smallest row.'''
def get_lexicographically_smallest_row(tableau, pivot_col_index):
    try:
        candidates = [] # tuple of row and index
        for r in range(1,tableau.shape[0]):
            if tableau[r,pivot_col_index]>0.0:
                row = tableau[r,:]/tableau[r,pivot_col_index]
                candidates.append([row, r])
        smallest_candidate_index = 0
        for i in range(1,len(candidates)):
            result = lexicographically_smaller(candidates[smallest_candidate_index][0], candidates[i][0])
            if result == 2:
                smallest_candidate_index = i
        return candidates[smallest_candidate_index][1]
    except:
        pass

def check_if_lexicographically_positive(fullTab):
    ''' first non zero element in every row except the first row should be positive'''
    flag = True
    for row in range(1,fullTab.shape[0]):
        for val in np.nditer(fullTab[row,:]):
            if val == 0.0:
                continue
            elif val > 0.0:
                break
            else:
                print("Row {} is negative".format(row))
                return False
    return flag

def get_basic_solutions(fullTab):
    ''' assuming basic indices are the only variables having zero reduced cost, will also check the column to see
    if it is a unit vector'''
    indices = []
    solution = []
    sum_along_columns = fullTab.sum(axis=0)

    for col in range(1,fullTab.shape[1]):
        if fullTab[0,col]==0.0 and sum_along_columns[col]==1.0:
            indices.append(col) # index of the basic variable
            for row in range(1,fullTab.shape[0]):
                if fullTab[row,col]==1.0:
                    solution.append(fullTab[row,0])
        else:
            solution.append(0.0)
    print("solution is \n{}".format(solution))
    return solution





''' first thing to do is to get data in the desired format. To begin with we assume that we have the initial bfs
A is full rank matrix, all vectors are cols by default.'''
A = np.array([[1,2,2,1,0,0],[2,1,2,0,1,0],[2,2,1,0,0,1]])
b = np.array([20,20,20])
c = np.array([-10,-12,-12,0,0,0])
x = np.array([0,0,0,20,20,20])

''' Below is text book example of Big M method, we introduced m dummy variables (as many as the number of constraints) 
and associated very large cost with them , initial basis is then just setting these dummy variables to b, and all 
other variables to be 0.'''

LARGE_VALUE = 999.0
A = np.array([[1,2,3,0,1,0,0,0],[-1,2,6,0,0,1,0,0],[0,4,9,0,0,0,1,0],[0,0,3,1,0,0,0,1]])
b = np.array([3,2,5,1])
c = np.array([1,1,1,0,LARGE_VALUE,LARGE_VALUE,LARGE_VALUE,LARGE_VALUE])
x = np.array([0,0,0,0,3,2,5,1])

A = np.load("P2_A.npy")
b = np.load("P2_b.npy")
c = np.load("P2_c.npy")
x = np.load("P2_bfs.npy")
print("shapes of inputs is A {}, b {}, c {}".format(A.shape, b.shape, c.shape))

''' form a full tableau matrix 
1. form the basis matrix
2. calculate the inverse of basis matrix
3. formulate C_B, assuming that the initial basic solution is not degenerate we can iterate over non zero values in 
x and that way we can identify the basic variable and hence we can identify basic columns.'''

'''below will give us rows, col indices of non zero elements in the vectors.'''

NZ = np.nonzero(x)
# print("Indices of non zero elements in the vector \n {}".format(NZ))

''' now we need to selectively choose the columns of A that would form the basis, which are the ones
corresponding to the basic variables'''

B = None

for x in np.nditer(NZ):
    col = A[:,x]
    '''match the dim, below will give a row vector, so need to transpose'''
    col = np.array([col]).T
    if B is None:
        B = col
    else:
        B = np.hstack((B,col))
# print("Here is the basis matrix in identity format. \n{}".format(B))
#----------------------------------------------------------------------------------
B = np.identity(36) #164, 20
#----------------------------------------------------------------------------------
'''need to calculate inverser of Basis matrix.'''
B_INV = inv(B)
# print("Here is the inverse of basis matrix. \n{}".format(B_INV))

''' now need to calculate c_b, costs corresponding to the basic variables'''
c_b_list = []
for x in np.nditer(NZ):
    c_b_list.append([c[x]])

c_b = np.array(c_b_list)
# print("cost coeffs of basic variables in are \n{}".format(c_b))

#------------------------------------ HACK, real problem is degeneracy in initial bfs-----
c_b_list = []
for i in range(0,36): #164, 20
    c_b_list.append([LARGE_VALUE])
c_b = np.array(c_b_list)
#------------------------------------------------------------------------------------------


'''B_INV * A'''
B_INV_A = np.dot(B_INV, A)
# print("B_INV * A is \n{}".format(B_INV_A))

''' B_INV * b'''
B_INV_b = np.dot(B_INV, b)
# print("B_INV * b is {}".format(B_INV_b))

''' -c_b.T * B_INV *b'''
neg_curr_cost = -1 * np.dot(np.transpose(c_b), B_INV_b)
print("neg current cost is \n{}".format(neg_curr_cost))

''' reduced costs row '''
reduced_costs = np.transpose(c) - np.dot(np.transpose(c_b), B_INV_A)
print("reduced costs row is {}".format(reduced_costs))


''' have all the pieces lets try to get the initial tableau '''
''' stacking reduced costs on top of B_INV_A '''
AUG_RIGHT = np.vstack((reduced_costs,B_INV_A))
# print("AUG_RIGHT is \n{}".format(AUG_RIGHT))


AUG_LEFT = np.array([(neg_curr_cost.tolist() +   B_INV_b.tolist())]).T
# print("AUG_LEFT is \n{}".format(AUG_LEFT))

FULL_TABLEAU = np.hstack((AUG_LEFT,AUG_RIGHT))
print("FULL_TABLEAU is \n{}".format(FULL_TABLEAU))
np.save("full_talbleau",FULL_TABLEAU)

'''----------------------------TOY DEGEN EXAMPLE FROM BOOK------------------------------------------------'''
# FULL_TABLEAU = np.array([[3,-.75,20,-.5,6,0,0,0],[0,.25,-8,-1,9,1,0,0],[0,.5,-12,-.5,3,0,1,0],[1,0,0,1,0,0,0,1]])

'''----------------------------TOY DEGEN EXAMPLE FROM BOOK------------------------------------------------'''



FULL_TABLEAU= np.around(FULL_TABLEAU,decimals=2)
curr_sol = get_basic_solutions(FULL_TABLEAU)
for val in curr_sol:
    if val != 0.0:
        print("initial bfs not all zeros\n")



# if not check_if_lexicographically_positive(FULL_TABLEAU):
#     print(" INITIAL TABLEAU IS NOT LEX POSITIVE !\n")
#     exit(0)

if np.linalg.matrix_rank(A) != A.shape[0]:
    print(" Rows of A are not INDEPENDENT \n")
    exit(0)


''' 
now I need to implement the algorithm for iterating over Full Tableau's
identify the pivot column, identify the pivot element , identify the multipliers
'''
iter = 0
while not check_termination_condition(FULL_TABLEAU):
    '''identify pivot column , look for negative values in Zeroth row'''
    iter +=1
    # if iter==600:
    #     break
    index = 0
    pivot_column_candidates = []
    zeroth_row = FULL_TABLEAU[0,:]
    for val in np.nditer(zeroth_row):
        if val<0.0 and index !=0:
            pivot_column_candidates.append(index)
        index +=1
    # print(" pivot column candidates are \n{}".format(pivot_column_candidates))

    ''' randomly pick a pivot column from the candidate list'''
    pivot_column_index = random.choice(pivot_column_candidates)
    ''' ----------------------------------blands rule-------------------------------------------------------------'''
    pivot_column_index = pivot_column_candidates[0]
    ''' ----------------------------------blands rule-------------------------------------------------------------'''

    print("pivot column index is \n{}".format(pivot_column_index))

    ''' now we have to select the pivot row.'''
    ''' identify all row indices candidates by looking at positive entries in pivot column'''
    pivot_column = FULL_TABLEAU[:,pivot_column_index]
    pivot_row_index = None
    index = 0
    ratio = 999999999.0
    for val in np.nditer(pivot_column):
        if val > 0.0 and index != 0:
            if (FULL_TABLEAU[index,0]/val < ratio):
                ratio = FULL_TABLEAU[index,0]/val
                pivot_row_index = index
            # print("ratio is {}\n".format(ratio))
        index +=1



    #-------------------------------------------------------------- Lexicographic--------------------------------------
    # pivot_row_index = get_lexicographically_smallest_row(FULL_TABLEAU, pivot_column_index)
    #-------------------------------------------------------------- Lexicographic--------------------------------------
    print("pivot row index is \n{}".format(pivot_row_index))

    ''' now we have the pivot element'''
    pivot_element = FULL_TABLEAU[pivot_row_index, pivot_column_index]
    print("pivot element is \n{}".format(pivot_element))

    ''' now next step is to construct the row operation matrix.'''
    ''' We start with a m x m identity matrix'''
    row_transformation_matrix = np.identity(FULL_TABLEAU.shape[0])

    for i in range(0,FULL_TABLEAU.shape[0]):
        if i!=pivot_row_index:
            multiplier = -pivot_column[i]/pivot_element
        else:
            multiplier = 1/pivot_element
        row_transformation_matrix[i,pivot_row_index]=multiplier

    # print(row_transformation_matrix)

    # print (np.dot(row_transformation_matrix, FULL_TABLEAU))
    OLD_TABLEAU = FULL_TABLEAU
    FULL_TABLEAU = np.dot(row_transformation_matrix, FULL_TABLEAU)
    FULL_TABLEAU = np.around(FULL_TABLEAU,decimals=6)
    if not check_if_lexicographically_positive(FULL_TABLEAU):
        print(" TABLEAU IS NOT LEX POSITIVE !\n")
    print("neg current cost is \n{0}".format(FULL_TABLEAU[0,0]))
    new_sol = get_basic_solutions(FULL_TABLEAU)
    ''' need to see if zeroth row is lexicographically larger with every iteration,
    so need to keep track of zeroth rows'''
    new_zeroth_row = FULL_TABLEAU[0,:]
    if lexicographically_smaller(new_zeroth_row, zeroth_row) != 2:
        if lexicographically_smaller(new_zeroth_row, zeroth_row)==0:
            print("OLD and NEW ZEROTH ROWS ARE THE SAME!\n")
        print("OLD ZEROTH ROW \n{}".format(zeroth_row))
        print("NEW ZEROTH ROW \n{}".format(new_zeroth_row))
        print("ZEROTH ROW NOT LEX LARGER !!!! \n")
    diff = sum([m - n for m,n in zip(new_sol,curr_sol)])
    if diff==0.0:
        print("---- CYCLING !!!! , CURRENT AND NEW SOLUTIONS ARE SAME------")
    curr_sol = new_sol




print("final tableau is \n{}".format(FULL_TABLEAU))
print("neg current cost is \n{0}".format(FULL_TABLEAU[0,0]))
get_basic_solutions(FULL_TABLEAU)


''' get all the basic feasible solutions.'''

indices = []
solution_row_col = {} # this row in zeroth column corresponds to the col in zeroth row (decision variable.)
sum_along_columns = FULL_TABLEAU.sum(axis=0)
for column_index in range(1,FULL_TABLEAU.shape[1]):
    # basic solutions have zero reduced cost and are columns of identity matrix.
    if sum_along_columns[column_index]==1.0:
        column = FULL_TABLEAU[:,column_index]
        row_index = 0
        for val in np.nditer(column):
            if val == 1.0:
                # this means that row_index in zeroth column corresponds to decision variable at column index at zeroth row.
                solution_row_col[row_index]=column_index
            row_index+=1

print(solution_row_col)

solutions = [0 for i in range(0,FULL_TABLEAU.shape[1])] # zeroth index is never used
for row_index in range(1,FULL_TABLEAU.shape[0]):
    solutions[solution_row_col[row_index]] = FULL_TABLEAU[row_index,0]
    row_index+=1

# first index is dummy just get rid of it
solutions = solutions[1:]

print("\n---------------SUMMARY---------------\n")
print("value of objective function is \n{}".format(-1 * FULL_TABLEAU[0,0]))
print("number of iterations required to reach final solution is \n{}".format(iter))
print("Final solution including theta+ve, theta-ve, slack and Big-M variables is as below\n {}".format(solutions))
''' To get the original solution we are only interested in theta+ve and theta-ve, which are first 18 and second 18
solutions respecitively'''
theta_positive_solutions = solutions[:18]
theta_negative_solutions = solutions[18:36]
x_solutions = []
for i in range(0,len(theta_positive_solutions)):
    x_solutions.append(theta_positive_solutions[i] - theta_negative_solutions[i])


print("Final theta+ve solutions are \n{}".format(theta_positive_solutions))
print("Final theta-ve solutions are \n{}".format(theta_negative_solutions))


print("Final solution in terms of original decison variables are {}\n".format(x_solutions))
scipy.io.savemat('Group_4_Solution_2.mat', dict(x=x_solutions))




