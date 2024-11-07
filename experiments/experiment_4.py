import sys
import os
from kernel_methods import KernelApprox
import numpy as np
import time
import pandas as pd
from py_markdown_table.markdown_table import markdown_table
import random
import sys
import scipy.io
from scipy.spatial.distance import cdist
from rpchol.matrix import KernelMatrix
from rpchol.rpcholesky import rpcholesky
from sklearn.kernel_approximation import Nystroem
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler




# Load the data and normalize it.
filename = "precipitation.txt"
data = pd.read_csv(filename)
matrix = data[['latitude', 'longitude', 'time']].values
scaler = StandardScaler()
normalized_matrix = scaler.fit_transform(matrix)



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def format_e(n):
    a = '%.3E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


# The name of the kernel.
kernel_names = ["squared-exponential"]
# The number of Chebyshev points to take
num_chebyshev_points = 32

# The number of samples taken from the parameter space
num_samples = 300
# Dimension of the problem
dimension = 3

X = normalized_matrix
Y = X


eprint("start")
radius = cdist(X, np.zeros((1, X.shape[1])))
radius = max(radius)
D_b = radius

# tensor train  lists.
online_time_tt_uncompress = []
online_time_tt_compress = []
rank_lst = []
errs_tt_compress = []
errs_tt_uncompress = []
# nystrom lists
errs_nystrom = []
online_nystrom = []

# RP-cholesky lists
errs_rp = []
online_rp = []

tol_lst = [1E-5]


param_ell = np.random.uniform(D_b * .4, D_b, num_samples)

print("Dimension of Problem: ", dimension)
print("Number of Chebyshev Points used: ", num_chebyshev_points)
print("Tolerance: ", tol_lst)

data = []

# generate random submatrix.
s_l = random.sample(range(X.shape[0]), 500)
s_r = s_l


for kernel_name in kernel_names:
    for my_tol in tol_lst:
        offline_time_tt = time.perf_counter()
        TTK = KernelApprox(X, Y, num_chebyshev_points)
        sto, Q_factor = TTK.parametric_kernel_approximation(kernel_name, [(D_b*.4, D_b)], "tt",
                                                      tol=my_tol)
        offline_time_tt = time.perf_counter() - offline_time_tt
        tt_uncompress_rank = Q_factor.shape[1]


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




            online_t0 = time.perf_counter()
            A = KernelMatrix(X, bandwidth=ell)
            Z = rpcholesky(A, Sh.shape[0])
            online_rp.append(time.perf_counter() - online_t0)
            Z = (Z.G).T
            K_approx = Z[s_l, :] @ Z.T[:, s_r]
            my_err_rp = TTK.evaluate_error(K_true, K_approx, 'fro')
            errs_rp.append(my_err_rp)

            sk_kernel = RBF(length_scale=ell / np.sqrt(2))
            online_t0 = time.perf_counter()
            ny_approx = Nystroem(sk_kernel, n_components=Sh.shape[0])
            Z = ny_approx.fit_transform(X)
            online_nystrom.append(time.perf_counter() - online_t0)
            K_approx = Z[s_l, :] @ Z.T[:, s_r]
            my_err_nystrom = TTK.evaluate_error(K_true, K_approx, 'fro')
            errs_nystrom.append(my_err_nystrom)
            eprint("done")




        dict_tt_un = {"Method": "PTTK-Global-1",
                      "Storage(Mb)": format_e(sto * 8E-6),
                      "Offline Time": format_e(offline_time_tt),
                      "Error(mean)": format_e(np.mean(errs_tt_uncompress)),
                      "Error(max)": format_e(np.max(errs_tt_uncompress)),
                      "Online Time": format_e(np.mean(online_time_tt_uncompress))
                }

        dict_tt_con = {"Method": "PTTK-Global-2",
                      "Storage(Mb)": format_e(sto * 8E-6),
                      "Offline Time": format_e(offline_time_tt),
                      "Error(mean)": format_e(np.mean(errs_tt_compress)),
                      "Error(max)": format_e(np.max(errs_tt_compress)),
                      "Online Time": format_e(np.mean(online_time_tt_compress))
                }

        dict_rp = {"Method": "RP-Cholesky",
                   "Storage(Mb)": 0,
                   "Offline Time": 0,
                    "Error(mean)": format_e(np.mean(errs_rp)),
                    "Error(max)": format_e(np.max(errs_rp)),
                    "Online Time": format_e(np.mean(online_rp)),
                }

        dict_ny = {"Method": "Nystrom",
                   "Storage(Mb)": 0,
                   "Offline Time": 0,
                   "Error(mean)": format_e(np.mean(errs_nystrom)),
                    "Error(max)": format_e(np.max(errs_nystrom)),
                    "Online Time": format_e(np.mean(online_nystrom)),
                }

        data.append(dict_tt_un)
        data.append(dict_tt_con)
        data.append(dict_rp)
        data.append(dict_ny)
        eprint("done")





markdown = markdown_table(data).get_markdown()
print(markdown)
print("The mean compression rank is: ", np.mean(rank_lst))
print("The uncompressed rank is: ", tt_uncompress_rank)
