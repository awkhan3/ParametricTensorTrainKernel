import sys
import os
from kernel_methods import KernelApprox
import numpy as np
import time
from py_markdown_table.markdown_table import markdown_table
from cross import aca_partial


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def format_e(n):
    a = '%.3E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


kernel_names = ["matern", "squared-exponential", "multiquadrics", "thin-plate-spline"]
# The number of Chebyshev points to take
num_chebyshev_points = 32
# The number of source and target points.
num_source_points = 5000
num_target_points = 5000
# The number of samples taken from the parameter space
num_samples = 300
# Dimension of the problem
dimension = 3

# generate source and target points
X = np.random.uniform(0, 1, (num_source_points, dimension))
Y = np.random.uniform(1, 2, (num_target_points, dimension))
D_b = np.sqrt(3)

# initialize low rank kernel approximation object
TTK = KernelApprox(X, Y, num_chebyshev_points)

# tensor train lists.
offline_time_tt = []
online_time_tt = []
errs_tt = []
storage_cost_tt = []
data_tt = []

# ACA lists
errs_ACA = []
time_ACA = []
data_aca = []

tol_lst = [1E-4, 1E-6, 1E-8]

compression_ranks = []

# generate parameter space
param_ell = np.random.uniform(D_b * .5, D_b, num_samples)
param_nu = np.random.uniform(.5, 3, num_samples)

print("Dimension of Problem: ", dimension)
print("Number of Chebyshev Points used: ", num_chebyshev_points)
print("Num of Source Points: ", num_source_points)
print("Num of Target Points: ", num_target_points)
print("Source Box: ", 0, 1)
print("Target Box: ", 1, 2)


for kernel_name in kernel_names:
    for my_tol in tol_lst:
        offline_t0 = time.perf_counter()

        if kernel_name == "matern":
            sto, Fsh, Fth = TTK.parametric_kernel_approximation(kernel_name, [(D_b * .5, D_b), (.5, 3)], "tt",
                                                                tol=my_tol)
        else:
            sto, Fsh, Fth = TTK.parametric_kernel_approximation(kernel_name, [(D_b * .5, D_b)], "tt",
                                                                tol=my_tol)
        offline_t = time.perf_counter() - offline_t0
        r_compr = 0
        vec_fun = TTK.get_curr_kernel_function()
        for i in range(num_samples):
            ell = param_ell[i]
            nu = param_nu[i]

            if kernel_name == "matern":
                ell = param_ell[i]
                nu = param_nu[i]
                my_param = [ell, nu]
            else:
                ell = param_ell[i]
                my_param = [ell]

            param_tuple = tuple(my_param)
            p_vec_fun = lambda diff_sq: vec_fun(diff_sq, *param_tuple)

            online_t0 = time.perf_counter()
            Mh = TTK.online_mode(my_param)
            online_time_tt.append(time.perf_counter() - online_t0)
            K_true = TTK.generate_true_parametric_kernel_mat(my_param)
            K_approx = Fsh @ Mh @ Fth
            r_compr = min(list(np.shape(Mh)))

            t3 = time.perf_counter()
            U, S = aca_partial(X, Y, my_tol, p_vec_fun)
            time_ACA.append(time.perf_counter() - t3)
            # r_compr_2 = U.shape[1]
            aca_err = TTK.evaluate_error(K_true, U @ S, 'fro')
            errs_ACA.append(aca_err)

            my_err_2 = TTK.evaluate_error(K_true, K_approx, 'fro')
            errs_tt.append(my_err_2)

        dict_tt = {"Kernel": kernel_name,
                   "Tol": format_e(my_tol),
                   "Offline Time (s)": format_e(offline_t),
                   "Storage(MB)": format_e(sto * 8E-6),
                   "Online Time (s)": format_e(np.mean(online_time_tt)),
                   "Error": format_e(max(errs_tt))
                   }

        dict_aca = {"Kernel": kernel_name,
                    "Tol": format_e(my_tol),
                    "Online Time (s)": format_e(np.mean(time_ACA)),
                    "Speed Up": np.mean(time_ACA) / np.mean(online_time_tt),
                    "Error": format_e(max(errs_ACA))}

        data_tt.append(dict_tt)
        data_aca.append(dict_aca)
        eprint("done")

        errs_tt = []
        errs_ACA = []
        time_ACA = []
        online_time_tt = []
        
print("\n" + "-"*40)
print("PTTK".center(40))
print("-"*40)
markdown = markdown_table(data_tt).get_markdown()
print(markdown)

print("\n" + "-"*40)
print("ACA".center(40))
print("-"*40)
markdown = markdown_table(data_aca).get_markdown()
print(markdown)
