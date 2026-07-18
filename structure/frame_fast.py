from structure.point import Point
from typing import List
import swiglpk as glp
from structure.others import isZero


# check if the two vectors are linearly independent
def linearInd(p: Point, q: Point) -> bool:
    dim = p.dim
    # check if there exists vector with all 0, i.e., [0, 0, ..., 0]
    p_zero = True
    q_zero = True
    ratio = None
    for i in range(dim):
        if not isZero(q.coord[i]):
            q_zero = False
            ratio = p.coord[i] / q.coord[i]
        if not isZero(p.coord[i]):
            p_zero = False
    if p_zero or q_zero:
        return True

    # both vectors are not zero vectors
    for i in range(dim):
        if p.coord[i] != 0 and isZero(q.coord[i]):
            return True
        elif not isZero(q.coord[i]) and not isZero(ratio - p.coord[i] / q.coord[i]):
            return True

    return False


# compute the rank of the gauss matrix
def gaussRank(P: List[Point]) -> int:
    dim = P[0].dim
    # build a matrix
    cp_matrix = []
    for pp in P:
        v = [pp.coord[j] for j in range(dim)]
        cp_matrix.append(v)
    while len(cp_matrix) < dim:
        v = [0] * dim
        cp_matrix.append(v)

    n = len(cp_matrix)
    d = len(cp_matrix[0])

    for i in range(d):
        # Search for maximum in this column
        maxEl = abs(cp_matrix[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(cp_matrix[k][i]) > maxEl:
                maxEl = abs(cp_matrix[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, d):
            tmp = cp_matrix[maxRow][k]
            cp_matrix[maxRow][k] = cp_matrix[i][k]
            cp_matrix[i][k] = tmp

        # Make all rows below this one 0 in current column
        if cp_matrix[i][i] != 0:
            for k in range(i + 1, n):
                c = -cp_matrix[k][i] / cp_matrix[i][i]
                for j in range(i, d):
                    if i == j:
                        cp_matrix[k][j] = 0
                    else:
                        cp_matrix[k][j] += c * cp_matrix[i][j]

    count = 0
    for i in range(n):
        allzero = all(isZero(a) for a in cp_matrix[i])
        if not allzero:
            count += 1
    return count


def solveLP(B, b):
    col_num = len(B) + 1
    row_num = b.dim

    # Initialize the mean point
    mean = Point(row_num, coord=[0.0] * row_num)
    for i in range(row_num):
        for point in B:
            mean.coord[i] += point.coord[i]
        mean.coord[i] /= float(len(B))

    # Create a new LP problem
    lp = glp.glp_create_prob()
    glp.glp_set_prob_name(lp, "solveLP")
    glp.glp_set_obj_dir(lp, glp.GLP_MIN)

    # Add row_num rows to the problem
    for i in range(1, row_num + 1):
        glp.glp_add_rows(lp, 1)
        glp.glp_set_row_name(lp, i, "q{i}")
        glp.glp_set_row_bnds(lp, i, glp.GLP_FX, b.coord[i - 1], b.coord[i - 1])

    # Add col_num columns to the problem
    for i in range(1, col_num + 1):
        glp.glp_add_cols(lp, 1)
        glp.glp_set_col_name(lp, i, "v{i}")
        glp.glp_set_col_bnds(lp, i, glp.GLP_LO, 0.0, 0.0)
        glp.glp_set_obj_coef(lp, i, 1.0 if i == 1 else 0.0)

    # Set the matrix of coefficients
    ia = glp.intArray(row_num * col_num + 1)
    ja = glp.intArray(row_num * col_num + 1)
    ar = glp.doubleArray(row_num * col_num + 1)
    count = 1
    for i in range(1, row_num + 1):
        for j in range(1, col_num + 1):
            ia[count] = i
            ja[count] = j
            if j == 1:
                ar[count] = -float(mean.coord[i - 1])
            else:
                ar[count] = B[j - 2].coord[i - 1]
            count += 1
    # Load the matrix into the LP problem
    glp.glp_load_matrix(lp, count - 1, ia, ja, ar)

    # Solve the problem using the simplex method
    smcp = glp.glp_smcp()
    glp.glp_init_smcp(smcp)
    smcp.msg_lev = glp.GLP_MSG_OFF
    glp.glp_simplex(lp, smcp)

    # Check if the problem is feasible.
    # feasible = glp.glp_get_status(lp) == glp.GLP_NOFEAS

    # Retrieve the solution
    theta = glp.glp_get_obj_val(lp)
    pi_coord = [glp.glp_get_row_dual(lp, i) for i in range(1, row_num + 1)]
    pi = Point(row_num, -1, pi_coord)

    # Clean up
    glp.glp_delete_prob(lp)

    return theta, pi


def findExtremeRay(R: List[Point], pi: Point, sigma: Point) -> Point:
    max_ratio = float('-inf')
    U = []

    for pp in R:
        if pi.dot_prod(pp) > 0:
            ratio = sigma.dot_prod(pp) / pi.dot_prod(pp)
            if ratio > max_ratio and not isZero(ratio - max_ratio):
                max_ratio = ratio
                U.clear()
                U.append(pp)
            elif isZero(ratio - max_ratio):
                U.append(pp)
    if len(U) <= 2:
        e = U[0]
    else:
        tmpP = [item for item in U]
        tmpB = frameConeFastLP(tmpP)
        e = tmpB[0]
    return e


def initial(rays: List[Point]) -> (int, List[Point], int):
    rank = gaussRank(rays)
    dim = rays[0].dim
    minus_ray = Point(dim)
    for i in range(dim):
        minus_ray.coord[i] = -rays[0].coord[i]
    theta, pi = solveLP(rays, minus_ray)

    rays_middle = []
    for pp in rays:
        if isZero(pi.dot_prod(pp)):
            rays_middle.append(pp)

    if len(rays_middle) == rank - 1:
        raysR = rays_middle
    else:
        tmpP = [item for item in rays_middle]
        tmpB = frameConeFastLP(tmpP)
        raysR = tmpB

    pi2 = Point(pi.dim, None, pi.coord)
    index = 1
    while not linearInd(pi, pi2):
        for i in range(dim):
            minus_ray.coord[i] = -rays[index].coord[i]
        theta, pi2 = solveLP(rays, minus_ray)
        index += 1
    sigma = Point(dim)
    for i in range(dim):
        sigma.coord[i] = 0.5 * (pi.coord[i] + pi2.coord[i])
    R = [item for item in rays if item not in raysR]
    minuspi = Point(dim)
    for i in range(dim):
        minuspi.coord[i] = -pi.coord[i]
    e = findExtremeRay(R, minuspi, sigma)
    raysR.append(e)

    return rank, raysR, sigma


def frameConeFastLP(rays: List[Point]) -> List[Point]:
    rank, raysR, sigma = initial(rays)
    R = [item for item in rays if item not in raysR]

    while len(R) > 0:
        b = R[0]
        theta, pi = solveLP(raysR, b)
        v = pi.dot_prod(b)
        if v > 0 and not isZero(v):
            e = findExtremeRay(R, pi, sigma)
            raysR.append(e)
            R.remove(e)
        else:
            R.remove(b)
    return raysR


#test
'''
p = Point(4)
p.id = 5
p.coord[0] = 0.111577; p.coord[1] = 0.224432; p.coord[2] = -0.420030; p.coord[3] = -0.263901
q = Point(4)
q.id = 10
q.coord[0] = -0.172520; q.coord[1] = -0.133716; q.coord[2] = -0.319878; q.coord[3] = 0.478123
o = Point(4)
o.id = 15
o.coord[0] = 0.304933; o.coord[1] = -0.049315; o.coord[2] = -0.215458; o.coord[3] = -0.285397
o1 = Point(4)
o1.id = 20
o1.coord[0] = 0.539634; o1.coord[1] = 0.008198; o1.coord[2] = -0.600949; o1.coord[3] = -0.369628
re = gaussRank([p, q, o, o1])
print(re)
minus = Point(4)
minus.coord[0] = -0.111577; minus.coord[1] = -0.224432; minus.coord[2] = 0.420030; minus.coord[3] = 0.263901
frameConeFastLP([p, q, o, o1])
'''