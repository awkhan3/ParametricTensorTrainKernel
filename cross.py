import random
import numpy as np
from numba import jit
from scipy.spatial.distance import cdist
rng = np.random.default_rng()


def greedy_cross(u, fun, tol, nswp):
    """
    greedy_cross computes the tensor train cross approximation of a tensor whose entries are defined by fun and
    the dimensions of the tensor are of the same value. The code here is based on
    https://github.com/oseledets/TT-Toolbox/blob/master/cross/greedy2_cross.m

    :param u: list that contains the dimensions of the tensor
    :param fun: function whose outputs are the entries of the tensor
    :param tol: A proxy to relative error of the tensor train cross approximation in the Chebyshev norm,
     heuristically computed via sampling
    :param nswp: the maximum number of left to right sweeps
    :return: list of tensor train cores which are three-dimensional numpy arrays
    """

    # ind_left and ind_right are a list of matrices, where ind_left[k]
    # contains a matrix of size (r x  k+1) where r corresponds to the rank
    # in the sense of the number of crosses that matrix contains.
    # ind_left[k] is I <=k and ind_right[k] is I > k
    # mid_inv is a list of 2d tuples where each tuple encodes the k'th middle inverse matrix
    assert (len(set(u)) == 1)
    ind_left = []
    ind_right = []
    mid_inv = []
    factors = []
    pre_factors = []
    dim = len(u)
    max_dx_lst = [0] * dim
    maxy = 0
    swp = 1
    flags = [False] * (dim - 1)
    truth_flags = [True] * (dim - 1)

    ind_left, ind_right, factors, mid_inv, u, fun, left_exl, right_exl = init_cross_approximation(ind_left,
                                                                                                  ind_right, factors,
                                                                                                  mid_inv,
                                                                                                  u, fun)
    ind_left_exl = left_exl
    ind_right_exl = right_exl[::-1]

    # move this to init cross approximation.
    for i in range(len(factors) - 1):
        (r1, n1, r2) = np.shape(factors[i])
        (r2, n2, r3) = np.shape(factors[i + 1])
        f1 = factors[i].reshape(r1 * n1, r2) @ mid_inv[i][0]
        f2 = mid_inv[i][1] @ factors[i + 1].reshape(r2, n2 * r3)
        pre_factors.append([f1, f2])

    ind_selector = 0

    while True:

        if flags == truth_flags and ind_selector >= dim - 1:
            print("|sweep|: ", swp, "|max_error|: ", max(max_dx_lst))
            return form_tensor(factors, mid_inv, fun)

        if swp >= nswp:
            print("|sweep|: ", swp, "|max_error|: ", max(max_dx_lst))
            return form_tensor(factors, mid_inv, fun)

        if ind_selector >= dim - 1:
            if swp % 10 == 0:
                print("|sweep|: ", swp, "|max_error|: ", max(max_dx_lst))
            ind_selector = 0
            swp = swp + 1

        status = 2
        if ind_selector - 1 >= 0:
            I_le_ind = ind_left[ind_selector - 1]
        else:
            I_le_ind = 0
            status = 1

        if ind_selector + 1 < dim - 1:
            I_gr_ind = ind_right[ind_selector + 1]
        else:
            I_gr_ind = 0
            status = 3

        # make sure the factors are rehaped properly before operating on them
        # e.g left should rn x r and right should nr x r

        left_factor = factors[ind_selector]
        right_factor = factors[ind_selector + 1]
        left_factor = np.reshape(left_factor, (left_factor.shape[0] * left_factor.shape[1], left_factor.shape[2]))
        right_factor = np.reshape(right_factor,
                                  (right_factor.shape[0], right_factor.shape[1] * right_factor.shape[2]))

        # we must form this recursively
        # yl = left_factor @ mid_inv[ind_selector][0]
        # yr = mid_inv[ind_selector][1] @ right_factor

        yl = pre_factors[ind_selector][0]
        yr = pre_factors[ind_selector][1]

        if yl.shape[0] != len(ind_left_exl[ind_selector]) and yr.shape[1] != len(ind_right_exl[ind_selector]):

            ik, jk1, err_diff, maxy_t = get_new_cross(fun, I_le_ind, I_gr_ind, yl, yr,
                                                      u[ind_selector], u[ind_selector + 1], status, tol,
                                                      ind_left_exl[ind_selector],
                                                      ind_right_exl[ind_selector])
            maxy = max([abs(maxy_t), abs(maxy)])
            max_dx_lst[ind_selector] = err_diff / maxy
        else:
            flags[ind_selector] = True
            ind_selector = ind_selector + 1
            continue

        if abs(max_dx_lst[ind_selector]) < tol:
            flags[ind_selector] = True
            ind_selector = ind_selector + 1
            continue
        else:
            flags[ind_selector] = False
            ind_left_exl[ind_selector].append(ik)
            ind_right_exl[ind_selector].append(jk1)

        match status:
            case 1:
                i2, j2 = np.unravel_index(jk1, (u[ind_selector + 1], I_gr_ind.shape[0]))
                new_cross_i = ik
                new_cross_j = np.concatenate(([i2], I_gr_ind[j2, :]))

            case 2:
                i1, j1 = np.unravel_index(ik, (I_le_ind.shape[0], u[ind_selector]))
                i2, j2 = np.unravel_index(jk1, (u[ind_selector + 1], I_gr_ind.shape[0]))
                new_cross_i = np.concatenate((I_le_ind[i1, :], [j1]))
                new_cross_j = np.concatenate(([i2], I_gr_ind[j2, :]))
            case 3:
                i1, j1 = np.unravel_index(ik, (I_le_ind.shape[0], u[ind_selector]))
                new_cross_i = np.concatenate((I_le_ind[i1, :], [j1]))
                new_cross_j = jk1

        # fix the dimensions of the new cross
        new_cross_i = new_cross_i.reshape(1, np.size(new_cross_i))
        new_cross_j = new_cross_j.reshape(1, np.size(new_cross_j))

        # add new crosses, by adding a new row.
        ind_left[ind_selector] = np.vstack([ind_left[ind_selector], new_cross_i])
        ind_right[ind_selector] = np.vstack([ind_right[ind_selector], new_cross_j])

        # update our factor matrices
        temp_l = np.arange(u[ind_selector])
        temp_l = temp_l.reshape(u[ind_selector], 1)

        if status == 1:
            rn = temp_l
        else:
            rn = array_mesh(I_le_ind, temp_l)

        rn = true_vector(rn, new_cross_j, fun, False)
        rn = rn.reshape(np.size(rn), 1)
        factors[ind_selector] = np.hstack([left_factor, rn])

        temp_r = np.arange(u[ind_selector + 1])
        temp_r = temp_r.reshape(u[ind_selector + 1], 1)
        if status == 3:
            nr = temp_r
        else:
            nr = array_mesh(temp_r, I_gr_ind, False)

        nr = true_vector(new_cross_i, nr, fun)
        nr = nr.reshape(1, np.size(nr))
        factors[ind_selector + 1] = np.vstack([right_factor, nr])

        # after updating the factor matrices, we now reset them
        # into the default (r, n, r) form.

        r1n, r2 = np.shape(factors[ind_selector])
        r2, r3n = np.shape(factors[ind_selector + 1])

        # now update the mid_inv using block matrix inversion.
        # we wish to recursively build up the mid_inv matrix.
        # analytically express it in terms of the LU factors.

        # make sure bt is a transposed vector, make sure all the dimensions are right.

        b = true_vector(ind_left[ind_selector], new_cross_j, fun, False)
        bt = true_vector(new_cross_i, ind_right[ind_selector], fun)

        b = b[:(np.size(b) - 1)]
        bt = bt[:(np.size(bt) - 1)]

        b = b.reshape(np.size(b), 1)
        bt = bt.reshape(1, np.size(bt))

        u1 = mid_inv[ind_selector][0]
        l1 = mid_inv[ind_selector][1]
        l3 = 1.0
        k_fac = autovecfun(fun,
                           np.array(np.append(new_cross_i, new_cross_j)).reshape(1, np.size(new_cross_j) + np.size(
                               new_cross_i))) - \
                (bt @ u1) @ (l1 @ b)

        u2 = (-1 / k_fac) * (u1 @ (l1 @ b))
        l2 = -1 * (bt @ u1) @ l1
        r = mid_inv[ind_selector][0].shape[0]
        u1n = np.zeros((r + 1, r + 1))
        l1n = np.zeros((r + 1, r + 1))
        u1n[:r, :r] = u1
        u1n[r, r] = 1 / k_fac
        shp1 = np.shape(u1n[0:r, r])
        shp2 = np.shape(l1n[r, 0:r])
        u1n[0:r, r] = u2.reshape(shp1)
        l1n[:r, :r] = l1
        l1n[r, 0:r] = l2.reshape(shp2)
        l1n[r, r] = 1.0
        mid_inv[ind_selector] = (u1n, l1n)

        pre_factors[ind_selector] = [
            np.hstack([pre_factors[ind_selector][0], factors[ind_selector] @ u1n[:, r].reshape(r + 1, 1)]),
            np.vstack([pre_factors[ind_selector][1], l1n[r, :].reshape(1, r + 1) @ factors[ind_selector + 1]])]

        if ind_selector - 1 >= 0:
            rn = rn.reshape(int(np.size(rn) / u[ind_selector]), u[ind_selector])
            rn = mid_inv[ind_selector - 1][1] @ rn
            rn = rn.reshape(np.size(rn), 1)
            pre_factors[ind_selector - 1][1] = pre_factors[ind_selector - 1][1].reshape(r1n, r2 - 1)
            pre_factors[ind_selector - 1][1] = np.hstack([pre_factors[ind_selector - 1][1], rn])
            pre_factors[ind_selector - 1][1] = pre_factors[ind_selector - 1][1].reshape(int(r1n / u[ind_selector]),
                                                                                        u[ind_selector + 1] * r2)

        if ind_selector + 1 < len(pre_factors):
            nr = nr.reshape(u[ind_selector + 1], int(np.size(nr) / u[ind_selector]))
            nr = nr @ mid_inv[ind_selector + 1][0]
            nr = nr.reshape(1, np.size(nr))
            pre_factors[ind_selector + 1][0] = pre_factors[ind_selector + 1][0].reshape(r2 - 1, r3n)
            pre_factors[ind_selector + 1][0] = np.vstack([pre_factors[ind_selector + 1][0], nr])
            pre_factors[ind_selector + 1][0] = pre_factors[ind_selector + 1][0].reshape(r2 * u[ind_selector + 1],
                                                                                        int(r3n / u[ind_selector]))

        factors[ind_selector] = np.reshape(factors[ind_selector], (int(r1n / u[ind_selector]),
                                                                   u[ind_selector], r2))
        factors[ind_selector + 1] = np.reshape(factors[ind_selector + 1],
                                               (r2, u[ind_selector + 1], int(r3n / u[ind_selector + 1])))

        # Now that the factors are changed, the positions of left_exl and right_exl must be updated as well.
        temp_left_ind = np.array(ind_left_exl[ind_selector])
        temp_right_ind = np.array(ind_right_exl[ind_selector])
        if ind_selector > 0:
            temp_right_ind = np.array(ind_right_exl[ind_selector - 1])
            temp_right_ind = temp_right_ind.reshape(np.size(temp_right_ind), 1)
            temp_right_ind = np.unravel_index(temp_right_ind, [u[ind_selector], r2 - 1])
            temp_right_ind = np.ravel_multi_index(temp_right_ind, [u[ind_selector], r2])
            ind_right_exl[ind_selector - 1] = list(np.squeeze(temp_right_ind))
        if ind_selector < dim - 2:
            temp_left_ind = np.array(ind_left_exl[ind_selector + 1])
            temp_left_ind = temp_left_ind.reshape(np.size(temp_left_ind), 1)
            temp_left_ind = np.unravel_index(temp_left_ind, [r2 - 1, u[ind_selector]])
            temp_left_ind = np.ravel_multi_index(temp_left_ind, [r2, u[ind_selector]])
            ind_left_exl[ind_selector + 1] = list(np.squeeze(temp_left_ind))

        ind_selector = ind_selector + 1


# obtain a new cross
# where L is I<=k-1 and R is I>k+1
# L and R are matricies encoding the sets
# yl and yr correspond to the current tt cores
# being operated on.
# j_prev corresponds to an index w.r.t. yr, flattened single from nr
# case 1 means we are operating on first core, case 2 means we are
# operating on middle cores, and case 3 means we are operating on last core.
# j_prev is used to index into meshed arrays.
def get_new_cross(fun, L, R, yl, yr, n1, n2, status, maxy, ind_left_exl, ind_right_exl):

    ind1 = np.arange(n1)
    ind1 = ind1.reshape(n1, 1)
    ind2 = np.arange(n2)
    ind2 = ind2.reshape(n2, 1)

    ind_left_exl = np.array(ind_left_exl, dtype=int)
    ind_right_exl = np.array(ind_right_exl, dtype=int)

    match status:
        case 1:
            left = ind1
            right = array_mesh(ind2, R, False)
        case 2:
            left = array_mesh(L, ind1)
            right = array_mesh(ind2, R, False)
        case 3:
            left = array_mesh(L, ind1)
            right = ind2

    err_diff, maxy, max_i, max_j = random_error_check(left, right, yl, yr, fun, maxy, ind_left_exl,
                                                      ind_right_exl)

    return max_i, max_j, err_diff, maxy


def autovecfun(fun, J, vec_flag=True):
    if vec_flag:
        return fun(J)

    s1 = np.shape(J)[0]
    y = np.zeros(s1)
    for i in range(s1):
        y[i] = fun(*tuple(J[i, :]))
    return y


# switch == true => rn
# switch == false => nr
def array_mesh(left, right, switch=True):
    if switch:
        r = left.shape[0]
        n = right.shape[0]
        left = np.repeat(left, n, axis=0)
        right = np.tile(right, (r, 1))
        return np.hstack([left, right])
    else:
        n = left.shape[0]
        r = right.shape[0]
        left = np.repeat(left, r, axis=0)
        right = np.tile(right, (n, 1))
        return np.hstack([left, right])


# method to concatenate np.arange(n) and the right/left set of crosses
# switch == true then left is fixed
# switch == false then right is fixed
# where ind is an array
def true_vector(L, R, fun, switch=True):
    if switch:
        ind = L
        ind = np.repeat(ind, R.shape[0], axis=0)
        return autovecfun(fun, np.hstack([ind, R]))
    else:
        ind = R
        ind = np.repeat(ind, L.shape[0], axis=0)
        return autovecfun(fun, np.hstack([L, ind]))


def form_tensor(factors, mid_inv, fun):
    p = (np.array([1]).reshape((1, 1)), np.array([1]).reshape((1, 1)))
    temp_mid_inv = mid_inv.copy()
    temp_mid_inv.insert(0, p)
    temp_mid_inv.append(p)
    cores = []

    for i in range(0, len(factors)):
        core = factors[i]
        r1, n, r2 = np.shape(factors[i])
        core = temp_mid_inv[i][1] @ np.reshape(core, (r1, n * r2))
        core = np.reshape(core, (n * r1, r2)) @ temp_mid_inv[i + 1][0]
        core = np.reshape(core, (r1, n, r2))
        cores.append(core)

    return cores


def random_error_check(left, right, yl, yr, fun, maxy, left_exl, right_exl):
    rl, nl = np.shape(yl)
    nr, rr = np.shape(yr)

    left_sample = np.arange(rl)
    right_sample = np.arange(rr)

    if len(left_exl) > 0:
        left_sample = np.delete(left_sample, left_exl)
    if len(right_exl) > 0:
        right_sample = np.delete(right_sample, right_exl)

    left_sample = np.array(left_sample)
    right_sample = np.array(right_sample)

    # Use Floyd's algorithm to construct the max(rl, rr) samples.
    # moved the rng method.
    num_samples = min(np.size(left_sample), np.size(right_sample))
    samples = rng.choice(np.size(left_sample) * np.size(right_sample), num_samples, replace=False)
    ind_mat = np.array(np.unravel_index(samples, [np.size(left_sample), np.size(right_sample)]))
    left_sample = left_sample[ind_mat[0, :]]
    right_sample = right_sample[ind_mat[1, :]]

    # left samples and right samples are arrays of sample of len nr

    ind_mat1 = np.concatenate((left[left_sample, :], right[right_sample, :]), axis=1)
    y_vec = autovecfun(fun, ind_mat1)
    z = np.einsum('ij, ij->i', yl[left_sample, :], np.transpose(yr[:, right_sample]), optimize=True)

    err2 = y_vec - z
    ind = np.argmax(np.abs(err2))
    # nrm2 = np.sum(np.square(y_vec))

    max_i = left_sample[ind]
    max_j = right_sample[ind]

    # randomly fix the column or row, and then maximize w.r.t it.
    if bool(random.getrandbits(1)):
        fixed_j_index = np.repeat(right[max_j, :].reshape(1, np.size(right[max_j, :])),
                                  left.shape[0], axis=0)
        ind_mat_fixed_j = np.concatenate((left, fixed_j_index), axis=1)
        y_vec1 = autovecfun(fun, ind_mat_fixed_j)
        z1 = yl @ yr[:, max_j]
        err21 = y_vec1 - z1
        # we don't want crosses we already got.
        err21[left_exl] = 0
        max_i = np.argmax(np.abs(err21))
        err_diff = abs(err21[max_i])
    else:
        fixed_i_index = np.repeat(left[max_i, :].reshape(1, np.size(left[max_i, :])),
                                  right.shape[0], axis=0)
        ind_mat_fixed_i = np.concatenate((fixed_i_index, right), axis=1)
        y_vec1 = autovecfun(fun, ind_mat_fixed_i)
        z1 = yl[max_i, :] @ yr
        err21 = y_vec1 - z1
        # we don't want crosses we already got.
        err21[right_exl] = 0
        max_j = np.argmax(np.abs(err21))
        err_diff = abs(err21[max_j])

    maxy2 = np.max(y_vec)
    maxy3 = np.max(y_vec1)
    maxy = max([abs(maxy), abs(maxy2), abs(maxy3)])

    return err_diff, maxy, max_i, max_j


# initialize our cross approximation with random crosses.
# ind_left, ind_right, factors, and mid_inv are a list of matrices
# fun is a lambda function
# u is a list of integers
# start this algorithm with unrestricted pivoting.
# ideally start the algorithm with the largest pivot in abs_value.
# make a procedure that fixes the initial index.
# Initialize this procedure with the maximum.
def init_cross_approximation(ind_left, ind_right, factors, mid_inv, u, fun):
    non_vec_fun = lambda X: fun(np.array(X).reshape(1, np.size(X)))
    dim = len(u)
    num_crosses = 2

    ind_left, ind_right, left_exl, right_exl = pre_condition(u, fun)

    for i in range(len(ind_left)):
        ind_left[i] = np.array(ind_left[i], dtype=int)
        ind_right[i] = np.array(ind_right[i], dtype=int)

    # initialize the mid_inv list
    for ind_selector in range(dim - 1):
        left_ind = ind_left[ind_selector]
        right_ind = ind_right[ind_selector]
        g = lambda i, j: non_vec_fun((list(left_ind[i][:]) + list(right_ind[j][:])))
        A_cross = np.zeros((left_ind.shape[0], right_ind.shape[0]))
        for i in range(num_crosses):
            for j in range(num_crosses):
                A_cross[i, j] = g(i, j)

   #     print(np.linalg.cond(A_cross))

        # analytically construct LU
        u11 = A_cross[0, 0]
        u12 = A_cross[0, 1]
        u22 = 1
        u21 = 0

        l11 = 1
        l12 = 0
        l21 = A_cross[1, 0] / A_cross[0, 0]
        l22 = A_cross[1, 1] - (A_cross[1, 0] * A_cross[0, 1]) / A_cross[0, 0]

        L = np.array([[l11, l12], [l21, l22]])
        U = np.array([[u11, u12], [u21, u22]])
        L_inv = twobytwo_inv(L)
        U_inv = twobytwo_inv(U)

        # L, U = lu(A_cross)
        mid_inv.append((U_inv, L_inv))

    # initialize the factors list
    nr = array_mesh(np.arange(u[0]).reshape(u[0], 1), ind_right[0], False)
    C1 = autovecfun(fun, nr)
    C1 = C1.reshape(1, u[0], num_crosses)
    rn = array_mesh(ind_left[dim - 2], np.arange(u[dim - 1]).reshape(u[dim - 1], 1))
    Cn = autovecfun(fun, rn)
    Cn = Cn.reshape(num_crosses, u[dim - 1], 1)
    factors.append(C1)
    for ind_selector in range(1, dim - 1):
        left_ind = ind_left[ind_selector - 1]
        right_ind = ind_right[ind_selector]
        C = np.zeros((num_crosses, u[ind_selector], num_crosses))
        for i in range(u[ind_selector]):
            for j in range(num_crosses):
                for k in range(num_crosses):
                    C[j, i, k] = non_vec_fun((list(left_ind[j, :]) + [i] + list(right_ind[k, :])))
        factors.append(C)
    factors.append(Cn)

    return ind_left, ind_right, factors, mid_inv, u, fun, left_exl, right_exl


def pre_condition(u, fun):
    dim = len(u)
    rand_index = np.random.randint([0] * (dim - 1), [max(u) - 1] * (dim - 1), (2, dim - 1))
    it_count = 1
    left_exl = []
    right_exl = []
    cond_lst_1 = []
    cond_lst_2 = []
    max_cond = 0
    ind_cond_lst = []
    p_ind_right = []

    while it_count <= 10:
        ind_left = []
        ind_right = []

        max_cond_1 = 0
        for ind_selector in range(0, dim - 1):

            if ind_selector > 0:
                curr_left_ind = array_mesh(ind_left[ind_selector - 1], np.arange(u[ind_selector]).reshape(
                    u[ind_selector], 1))
                curr_left_ind = np.array(curr_left_ind, dtype=int)

            else:
                curr_left_ind = np.arange(u[ind_selector]).reshape(u[ind_selector], 1)
                curr_left_ind = np.array(curr_left_ind, dtype=int)

            m0 = true_vector(curr_left_ind, rand_index[0, ind_selector:].reshape(1, rand_index[0, ind_selector:].size),
                             fun, switch=False)
            m1 = true_vector(curr_left_ind, rand_index[1, ind_selector:].reshape(1, rand_index[1, ind_selector:].size),
                             fun, switch=False)

            cond_ind = fast_cond(m0, m1)
            if ind_selector == 0:
                cond_ind[np.diag_indices_from(cond_ind)] = np.inf
            i_min, j_min = np.unravel_index(cond_ind.argmin(), cond_ind.shape)
            left_exl.append([i_min, j_min])

            Z = np.zeros((2, curr_left_ind.shape[1]))
            Z[0, :] = np.array(curr_left_ind[i_min, :], dtype=int)
            Z[1, :] = np.array(curr_left_ind[j_min, :], dtype=int)
            ind_left.append(Z)

            Z = np.zeros((2, np.size(rand_index[0, ind_selector:])))
            Z[0, :] = np.array(rand_index[0, ind_selector:], dtype=int)
            Z[1, :] = np.array(rand_index[1, ind_selector:], dtype=int)
            p_ind_right.append(Z)

      #      print(cond_ind[i_min, j_min])
            max_cond_1 = max([max_cond_1, cond_ind[i_min, j_min]])
            cond_lst_1.append(cond_ind[i_min, j_min])
        ind_cond_lst.append((ind_left, p_ind_right, max_cond_1))
        p_ind_right = []

        max_cond_2 = 0
        for ind_selector in reversed(range(0, dim - 1)):
            if ind_selector == dim - 2:
                curr_right_ind = np.arange(u[ind_selector], dtype=int).reshape(u[ind_selector], 1)
            else:
                curr_right_ind = array_mesh(np.arange(u[ind_selector]).reshape(u[ind_selector], 1),
                                            ind_right[dim - 3 - ind_selector], False)
                curr_right_ind = np.array(curr_right_ind, dtype=int)

            curr_left_ind = ind_left[ind_selector]
            curr_left_ind = np.array(curr_left_ind, dtype=int)

            m0 = true_vector(curr_left_ind[0, :].reshape(1, curr_left_ind[0, :].size), curr_right_ind, fun)
            m1 = true_vector(curr_left_ind[1, :].reshape(1, curr_left_ind[1, :].size), curr_right_ind, fun)
            cond_ind = fast_cond(m0, m1)
            if ind_selector == dim - 2:
                cond_ind[np.diag_indices_from(cond_ind)] = np.inf
            i_min, j_min = np.unravel_index(cond_ind.argmin(), cond_ind.shape)
            right_exl.append([i_min, j_min])
            cond_lst_2.append(cond_ind[i_min, j_min])
            max_cond = max([max_cond, cond_ind[i_min, j_min]])
            max_cond_2 = max([max_cond_2, cond_ind[i_min, j_min]])

       #     print(cond_ind[i_min, j_min])

            Z = np.zeros((2, curr_right_ind.shape[1]))
            Z[0, :] = np.array(curr_right_ind[i_min, :], dtype=int)
            Z[1, :] = np.array(curr_right_ind[j_min, :], dtype=int)
            ind_right.append(Z)
        ind_cond_lst.append((ind_left, ind_right[::-1], max_cond_2))

        it_count = it_count + 1
        ind_right = ind_right[::-1]

        rand_index = np.array(ind_right[0], dtype=int)
    min_value = min(ind_cond_lst, key=lambda ind_cond_lst: ind_cond_lst[2])
    ind_left = min_value[0]
    ind_right = min_value[1]

    return ind_left, ind_right, left_exl, right_exl


@jit(nopython=True)
def fast_cond(m0, m1):
    cond_ind = np.zeros((m0.shape[0], m0.shape[0]))
    for i in range(m0.shape[0]):
        for j in range(m0.shape[0]):
            A = np.array([[m0[i], m1[i]],
                          [m0[j], m1[j]]])
            cond_ind[i, j] = np.linalg.cond(A)
    return cond_ind



def aca_partial(source_points, trg_points, tol, func):
    """
    aca_partial computes the matrix cross approximation of the kernel matrix defined by the isotropic
    kernel function func, source points (source_points), and target points (trg_points)
    such that the CUR decomposition ideally has a relative error less than or equal to the variable tol
    in the Frobenius norm. For more details, see Algorithm 1 in
    "Y. Liu, W. Sid-Lakhdar, E. Rebrova, P. Ghysels, and X.-S. Li.
    A parallel hierarchical blocked adaptive cross approximation algorithm.
    The International Journal of High Performance Computing Applications, 34(4):394–408, 2020."

    :param source_points: numpy matrix of size N_s times d where N_s is the number of source points and d is the dim
    of the source_points
    :param trg_points: numpy matrix of size N_t times d where N_t is the number of target points and d is the dim
    of the target points
    :param tol: tolerance of the relative error of the kernel matrix cross approximation in the Frobenius norm
    :param func: isotropic kernel function of the form func:R -> R st func has the form func(\|x-y\|_{2}^{2})
    :return: numpy matrices U, V of dimension N_s \times r and N_t \times r, respectively, such that ideally the
    relative error in the frobenius norm of the approximation is less than or equal to the variable tol
    """
    p, q = source_points.shape[0], trg_points.shape[0]
    rank = 10
    U = np.zeros((p, rank))
    V = np.zeros((rank, q))
    nrm2 = 0
    jk = random.randint(0, q - 1)
    k = 0
    while True:
        A_eval_at_col_jk = func(cdist(source_points, trg_points[jk, :].reshape(1, trg_points.shape[1]),
                                      metric='sqeuclidean'))
        uk = np.squeeze(A_eval_at_col_jk) - U @ V[:, jk]
        ik = np.argmax(np.abs(uk))
        A_eval_at_row_ik = func(cdist(source_points[ik, :].reshape(1, source_points.shape[1]), trg_points,
                                      metric='sqeuclidean'))
        vk = np.squeeze(A_eval_at_row_ik) - U[ik, :] @ V
        vk = vk / vk[jk]
        abs_vk = np.abs(vk)
        abs_vk[jk] = -1
        jk = np.argmax(abs_vk)
        nrm2 = nrm2 + 2 * np.dot(U.T @ uk, V @ vk) + (np.linalg.norm(uk) * np.linalg.norm(vk)) ** 2
        err2 = (np.linalg.norm(uk) * np.linalg.norm(vk)) ** 2
        U[:, k] = uk
        V[k, :] = vk
        if np.sqrt(err2) <= tol * np.sqrt(nrm2):
            return U[:, :k], V[:k, :]
        else:
            k = k + 1
            if k == rank:
                new_rank = min(2 * rank, p, q)
                tmp_U = np.zeros((q, new_rank))
                tmp_V = np.zeros((new_rank, p))
                tmp_U[:, :rank] = U
                tmp_V[:rank] = V
                U = tmp_U
                V = tmp_V
                rank = new_rank


def twobytwo_inv(A_cross):
    A_cross_inv = np.array([[A_cross[1, 1], -1 * A_cross[0, 1]], [-1 * A_cross[1, 0], A_cross[0, 0]]])
    A_cross_inv = (1 / (A_cross[1, 1] * A_cross[0, 0] - A_cross[0, 1] * A_cross[1, 0])) * A_cross_inv
    return A_cross_inv
