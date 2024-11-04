import sys
import os
from kernel_methods import KernelApprox
import numpy as np
import time
from py_markdown_table.markdown_table import markdown_table

def format_e(n):
    a = '%.3E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


kernel_names = ["exponential", "thin-plate", "bi-harmonic", "multiquadrics", "thin-plate-spline",
                "laplace2D", "laplace3D", "matern-3/2", "matern-5/2", "squared-exponential"]

# The number of Chebyshev points to take
num_chebyshev_points = 27
# The number of source and target points.
num_source_points = 10000
num_target_points = 10000
# dimension of the problem
dimension = 3
# generate the source and target points
X = np.random.uniform(0, 1, (num_source_points, dimension))
Y = np.random.uniform(2, 3, (num_target_points, dimension))
offline_t = time.perf_counter()
# form the low-rank kernel approximation objects
TTK = KernelApprox(X, Y, num_chebyshev_points)
offline_t = time.perf_counter() - offline_t
my_tol = 1E-9
data1 = []

print("Dimension of Problem: ", dimension)
print("Number of Chebyshev Points used: ", num_chebyshev_points)
print("Num of Source Points: ", num_source_points)
print("Num of Target Points: ", num_target_points)
print("Source Box: [0, 1]")
print("Target Box: [2, 3]")

for i in range(len(kernel_names)):
    kernel_name = kernel_names[i]
    TTK.kernel_approximation(kernel_name, tol=my_tol, method="tt")
    K_true = TTK.generate_true_kernel_mat()
    t0 = time.perf_counter()
    Fsh1, Fth1 = TTK.kernel_approximation(kernel_name, tol=my_tol, method="tt")
    t1 = time.perf_counter() - t0
    K_approx1 = Fsh1 @ Fth1
    err1 = TTK.evaluate_error(K_true, K_approx1, 2)
    t0 = time.perf_counter()
    U, S, Vh = np.linalg.svd(K_true, full_matrices=False)
    t2 = time.perf_counter() - t0
    S = np.diag(S)
    r_compr1 = np.shape(Fsh1)[1]
    svd_err1 = TTK.evaluate_error(K_true, U[:, :r_compr1] @ S[:r_compr1, :r_compr1] @ Vh[:r_compr1, :], 2)
    dict1 = {"Kernel": kernel_name,
             "Method": "TTK",
             "Tol": format_e(my_tol),
             "TTK-error": format_e(err1),
             "SVD-error": format_e(svd_err1),
             "Rank": r_compr1,
             "TTK-time": format_e(t1),
             "SVD-time": format_e(t2)}
    data1.append(dict1)
    print('done', file=sys.stderr)

markdown1 = markdown_table(data1).get_markdown()
print(markdown1)
