import sys
import os
from kernel_methods import KernelApprox
import numpy as np
import time
from py_markdown_table.markdown_table import markdown_table
from cross import aca_partial
import random


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def format_e(n):
    a = '%.3E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

# The name of the kernel.
kernel_names = ["squared-exponential", "multiquadrics"]
# The number of Chebyshev points to take
num_chebyshev_points = 27
# The number of source and target points.
num_source_points = 10**5
num_target_points = 10**5
# The number of samples taken from the parameter space
num_samples = 300
# Dimension of the problem
dimension = 3

X = np.random.uniform(0, 1, (num_source_points, dimension))
Y = X

D_b = np.sqrt(3)
# tensor train lists.
online_time_tt_uncompress = []
errs_tt_uncompress = []
online_time_tt_compress = []
errs_tt_compress = []
storage_cost_tt = []
rank_lst = []
uncompressed_rank = 0

tol_lst = [1E-5]

param_ell = np.random.uniform(D_b*.2, D_b, num_samples)

print("Dimension of Problem: ", dimension)
print("Number of Chebyshev Points used: ", num_chebyshev_points)
print("Num of Source Points: ", num_source_points)
print("Num of Target Points: ", num_target_points)
print("Source Box: ", 0, 1)
print("Target Box: ", 0, 1)
print("Tolerance: ", tol_lst)

data_uncompressed = []
data_compressed = []
# remember to define this in the paper.
# generate random sub-matrix.
s_l = random.sample(range(num_source_points), 500)
s_r = s_l

for kernel_name in kernel_names:
    for my_tol in tol_lst:
        offline_t = time.perf_counter()
        TTK = KernelApprox(X, Y, num_chebyshev_points)
        sto, Q_factor = TTK.parametric_kernel_approximation(kernel_name,  [(D_b*.2, D_b)], 'tt',
                                                            tol=my_tol)
        offline_t = time.perf_counter() - offline_t
        uncompressed_rank = Q_factor.shape[1]

        for i in range(num_samples):
            ell = param_ell[i]
            my_param = [ell]
            K_true = TTK.generate_true_parametric_kernel_mat(my_param, s_l=s_l, s_r=s_r)

            online_t = time.perf_counter()
            V, S = TTK.online_mode(my_param, compress_flag=False)
            online_t = time.perf_counter() - online_t
            online_time_tt_uncompress.append(online_t)

            online_t = time.perf_counter()
            Vh, Sh = TTK.online_mode(my_param, compress_flag=True)
            Q_factor_reduced = Q_factor @ Vh
            online_t = time.perf_counter() - online_t
            online_time_tt_compress.append(online_t)

            rank_lst.append(Sh.shape[0])

            K_approx = Q_factor[s_l, :] @ V @ S  @ V.T @ Q_factor[s_l, :].T
            K_approx_reduced = Q_factor_reduced[s_l, :] @ Sh @ Q_factor_reduced[s_l, :].T


            my_err_compress = TTK.evaluate_error(K_true, K_approx_reduced, 'fro')
            my_err_uncompress = TTK.evaluate_error(K_true, K_approx, 'fro')
            errs_tt_compress.append(my_err_compress)
            errs_tt_uncompress.append(my_err_uncompress)




        dict_uncompressed = {"Kernel": kernel_name,
                             "Offline Time (s)": format_e(offline_t),
                             "Error": format_e(np.mean(errs_tt_uncompress)),
                             "Storage (Mb)": format_e(sto*8*1E-6),
                             "Online Time (s)": format_e(np.mean(online_time_tt_uncompress)),
                             "Rank": uncompressed_rank}

        dict_compressed = {"Kernel": kernel_name,
                             "Offline Time (s)": format_e(offline_t),
                             "Error": format_e(np.mean(errs_tt_compress)),
                             "Storage (Mb)": format_e(sto*8*1E-6),
                             "Online Time (s)": format_e(np.mean(online_time_tt_compress)),
                             "Rank": np.mean(rank_lst)}

        data_uncompressed.append(dict_uncompressed)
        data_compressed.append(dict_compressed)

        errs_tt_uncompress = []
        errs_tt_compress = []
        online_time_tt_compress = []
        online_time_tt_uncompress = []
        rank_lst = []


        eprint("done")



print("\n" + "-"*40)
print("PTTK-Global-1".center(40))
print("-"*40)
markdown = markdown_table(data_uncompressed).get_markdown()
print(markdown)

print("\n" + "-"*40)
print("PTTK-Global-2".center(40))
print("-"*40)
markdown = markdown_table(data_compressed).get_markdown()
print(markdown)

