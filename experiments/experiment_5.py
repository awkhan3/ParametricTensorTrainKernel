from kernel_methods import KernelApprox
import numpy as np
import time
from py_markdown_table.markdown_table import markdown_table


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def format_e(n):
    a = '%.3E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


# the names of the kernel
kernel_names = ["exponential", "matern"]
# the number of Chebyshev points to take
num_chebyshev_points = 12
# the number of source and target points.
num_source_points = 5000
num_target_points = 5000
# the number of samples taken from the parameter space
num_samples = 300
# dimension of the problem
dimension = 3

# generate the source and target points
X = np.random.uniform(0, 1, (num_source_points, dimension))
Y = np.random.uniform(2, 3, (num_target_points, dimension))
D_b = 2 * np.sqrt(dimension)

# form the low-rank kernel approximation objects
TTK = KernelApprox(X, Y, num_chebyshev_points)
TTK_tuck = KernelApprox(X, Y, num_chebyshev_points)

# tensor train lists.
offline_time_tt = []
online_time_tt = []
errs_tt = []
storage_cost_tt = []
kernel_evals_tt = []
compression_ranks_tt = []
data_tt = []

# Tucker lists
errs_tuck = []
offline_time_tuck = []
online_time_tuck = []
kernel_evals_tuck = []
storage_cost_tuck = []
compression_ranks_tuck = []
data_tuck = []

tol_lst = [1E-3, 1E-4, 1E-8]

# generate the parameter spaces
param_ell = np.random.uniform(D_b * .5, D_b, num_samples)
param_nu = np.random.uniform(.5, 3, num_samples)

# print out problem setup parameters
print("Dimension of Problem: ", dimension)
print("Number of Chebyshev Points used: ", num_chebyshev_points)
print("Num of Source Points: ", num_source_points)
print("Num of Target Points: ", num_target_points)
print("Source Box: ", 0, 1)
print("Target Box: ", 2, 3)
print("D_b: ", D_b)

for kernel_name in kernel_names:
    for my_tol in tol_lst:

        if kernel_name == "matern":
            offline_time_t0 = time.perf_counter()
            sto, Fsh, Fth = TTK.parametric_kernel_approximation(kernel_name, [(D_b * .5, D_b), (.5, 3)], 'tt',
                                                                tol=my_tol)
            offline_time_tt.append(time.perf_counter() - offline_time_t0)

            offline_time_t0 = time.perf_counter()
            sto_tuck, Fsh_tuck, Fth_tuck = TTK_tuck.parametric_kernel_approximation(kernel_name,
                                                                                    [(D_b * .5, D_b), (.5, 3)],
                                                                                    'tucker',
                                                                                    ell=int(Fsh.shape[1] ** (
                                                                                            1 / dimension)) + 1)
            offline_time_tuck.append(time.perf_counter() - offline_time_t0)
            kernel_evals_tuck.append(num_chebyshev_points ** (2 * dimension + 2))

        else:
            offline_time_t0 = time.perf_counter()
            sto, Fsh, Fth = TTK.parametric_kernel_approximation(kernel_name, [(D_b * .5, D_b)], 'tt',
                                                                tol=my_tol)
            offline_time_tt.append(time.perf_counter() - offline_time_t0)

            offline_time_t0 = time.perf_counter()
            sto_tuck, Fsh_tuck, Fth_tuck = TTK_tuck.parametric_kernel_approximation(kernel_name, [(D_b * .5, D_b)],
                                                                                    'tucker',
                                                                                    ell=int(Fsh.shape[1] ** (
                                                                                            1 / dimension)) + 1)
            offline_time_tuck.append(time.perf_counter() - offline_time_t0)
            kernel_evals_tuck.append(num_chebyshev_points ** (2 * dimension + 1))

        vec_fun = TTK.get_curr_kernel_function()
        kernel_evals_tt.append(TTK.num_ker_evals)
        storage_cost_tuck.append(sto_tuck)
        storage_cost_tt.append(sto)
        compression_ranks_tt.append(np.max([Fsh.shape[1], Fth.shape[0]]))
        compression_ranks_tuck.append(np.max([Fsh_tuck.shape[1], Fth_tuck.shape[1]]))

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
            param_vec_fun = lambda diff_sq: vec_fun(diff_sq, *param_tuple)

            online_t0 = time.perf_counter()
            Mh = TTK.online_mode(my_param)
            online_time_tt.append(time.perf_counter() - online_t0)

            online_t0 = time.perf_counter()
            Mh_tuck = TTK_tuck.online_mode(my_param)
            online_time_tuck.append(time.perf_counter() - online_t0)

            K_true = TTK.generate_true_parametric_kernel_mat(my_param)
            K_approx = Fsh @ Mh @ Fth
            K_approx_tuck = Fsh_tuck @ Mh_tuck @ Fth_tuck.T
            r_compr = min(list(np.shape(Mh)))

            my_err_tt = TTK.evaluate_error(K_true, K_approx, 'fro')
            my_err_tuck = TTK.evaluate_error(K_true, K_approx_tuck, 'fro')
            errs_tt.append(my_err_tt)
            errs_tuck.append(my_err_tuck)

        dict_tt = {"Kernel": kernel_name, "Tol": format_e(my_tol),
                   "Offline Time (s)": format_e(offline_time_tt[-1]),
                   "Storage(MB)": format_e(storage_cost_tt[-1] * 8E-6),
                   "Online Time (s)": format_e(np.mean(online_time_tt)),
                   "Kernel Evals": kernel_evals_tt[-1],
                   "Rank": compression_ranks_tt[-1],
                   "Error": format_e(np.mean(errs_tt)),
                   }

        dict_tuck = {"Kernel": kernel_name, "Tol": format_e(my_tol),
                     "Offline Time": format_e(offline_time_tuck[-1]),
                     "Storage(MB)": format_e(storage_cost_tuck[-1] * 8E-6),
                     "Online Time": format_e(np.mean(online_time_tuck)),
                     "Kernel Evals": kernel_evals_tuck[-1],
                     "Rank": compression_ranks_tuck[-1],
                     "Error": format_e(np.mean(errs_tuck)),
                     }

        data_tt.append(dict_tt)
        data_tuck.append(dict_tuck)
        eprint("done")

        errs_tt = []
        errs_tuck = []
        online_time_tuck = []
        online_time_tt = []

print("\n" + "-" * 40)
print("PTTK".center(40))
print("-" * 40)
markdown = markdown_table(data_tt).get_markdown()
print(markdown)
print("\n" + "-" * 40)
print("Parametric Tucker".center(40))
print("-" * 40)
markdown = markdown_table(data_tuck).get_markdown()
print(markdown)


