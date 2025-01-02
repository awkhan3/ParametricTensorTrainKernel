import numpy as np
import scipy.linalg
import torch
from scipy.special import gamma, kv
import tensorly as tl
import tntorch as tn
from cross import greedy_cross
from scipy.spatial.distance import cdist
from numba import jit
import scipy as sci


class KernelApprox:
    """
    KernelApprox is a universal driver class used to obtain a low rank approximation of a kernel matrix using
    the tensor decompositions TT and Tucker.
    :param source_matrix: is a matrix of dimension N_s x spatial_dim, where N_s is the number of source points
    :param target_matrix: is a matrix of dimension N_t x spatial_dim, where N_t is the number of target points
    :param num_c_points: indicates the number of chebyshev points
    :return: KernelApprox class object
    """

    def __init__(self, source_matrix, target_matrix, num_c_points):
        assert source_matrix.ndim == 2
        assert target_matrix.ndim == 2
        assert source_matrix.shape[1] == target_matrix.shape[1]
        assert num_c_points > 3

        self.source_matrix = source_matrix
        self.target_matrix = target_matrix
        self.num_c_points = num_c_points
        self.source_poly_lst = []
        self.target_poly_lst = []
        self.source_c_points = []
        self.target_c_points = []
        self.source_c_matrix = np.zeros((num_c_points, source_matrix.shape[1]))
        self.target_c_matrix = np.zeros((num_c_points, target_matrix.shape[1]))
        self.param_c_matrix = None
        self.spatial_dimension = int(source_matrix.shape[1]) + int(target_matrix.shape[1])
        self.param_flag = False
        self.ttk_instance = None
        self.curr_kernel_func = None
        self.method = None
        self.num_ker_evals = 0
        self.kernel_names = ["exponential", "squared-exponential", "thin-plate", "bi-harmonic", "matern",
                             "multiquadrics", "thin-plate-spline", "laplace2D", "laplace3D", "matern-3/2",
                             "matern-5/2"]

        d = int(self.spatial_dimension / 2)
        for i in range(d):
            a = min(source_matrix[:, i])
            b = max(source_matrix[:, i])
            poly = InterpolatingPolynomial(a, b, num_c_points)
            self.source_c_matrix[:, i] = np.array(poly.get_nodes())
            self.source_poly_lst.append(poly)
            self.source_c_points.append(poly.get_nodes())

        for i in range(d):
            a = min(target_matrix[:, i])
            b = max(target_matrix[:, i])
            poly = InterpolatingPolynomial(a, b, num_c_points)
            self.target_c_matrix[:, i] = np.array(poly.get_nodes())
            self.target_c_points.append(poly.get_nodes())
            self.target_poly_lst.append(poly)

    def kernel_approximation(self, kernel_name, ell=1, nu=.5, method="tt", tol=1E-2, rnk=2):
        """
        kernel_approximation is a public method that computes a low-rank approximation of a non-parametric kernel matrix
        using TTK or Tucker.
        :param kernel_name: is a string containing the name of the kernel, supported kernels are  ["exponential",
        "squared-exponential", "thin-plate", "bi-harmonic","matern", "multiquadrics", "thin-plate-spline", "laplace2D",
        "laplace3D", "matern-3/2", "matern-5/2"]
        :param ell: the length scale parameter
        :param nu: the nu parameter associated with the Matérn kernel.
        :param method: for TTK give value "tt"; otherwise, give "tucker" for Tucker.
        :param tol: The tolerance given to the tensor-train cross approximation method
        :param rnk: The rank given to the HOOI method used to compute the Tucker decomposition of the tensor
        :return: low-rank approximation of the kernel matrix in the form [S, T] where S and T are numpy matrices
        and the low-rank approximation is S*np.transpose(T)
        """
        assert 1E-16 <= tol <= 1E-1
        assert kernel_name in self.kernel_names
        self.param_flag = False
        self.method = method

        # Change this match case semantic to a dict when kernel options get large.
        match kernel_name:
            case "matern":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._matern_kernel(diff_sq, ell, nu)
            case "squared-exponential":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._squared_exponential(diff_sq, ell)
            case "exponential":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._exponential(diff_sq, ell)
            case "bi-harmonic":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._bi_harmonic(diff_sq)
            case "thin-plate":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._thin_plate(diff_sq)
            case "multiquadrics":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._multiquadrics(diff_sq, ell)
            case "thin-plate-spline":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._thin_plate_spline(diff_sq, ell)
            case "laplace2D":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._laplace2D(diff_sq)
            case "laplace3D":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._laplace3D(diff_sq)
            case "matern-3/2":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._matern_kernel(diff_sq, ell, 3 / 2)
            case "matern-5/2":
                self.curr_kernel_func = lambda diff_sq: KernelApprox._matern_kernel(diff_sq, ell, 5 / 2)

        my_vec_ker = lambda M: self._vec_kernel(M, self.curr_kernel_func, 0)
        if method == "tt":
            self.ttk_instance = _TensorTrainKernel(self.source_matrix, self.target_matrix, self.source_poly_lst,
                                                   self.target_poly_lst, my_vec_ker, self.num_c_points)
            return self.ttk_instance.offline_mode(tol)
        elif method == "tucker":
            self.ttk_instance = Tucker_Kernel_Approx(self.source_matrix, self.target_matrix, self.source_poly_lst,
                                                     self.target_poly_lst, self.curr_kernel_func, rnk,
                                                     self.num_c_points)
            return self.ttk_instance.offline_mode()

    def parametric_kernel_approximation(self, kernel_name, params, method, tol=1E-1, ell=5):
        """
        parametric_kernel_approximation is a public method that performs the offline phase of PTTK or parametric Tucker
        depending on the string passed to method
        :param kernel_name: is a string containing the name of the kernel, supported kernels are ["thin-plate-spline",
        "matern", "squared-exponential", "exponential", "multiquadrics", "matern-5/2", "matern-3/2"]
        :param params: is the parameter space of the parametric kernel, encoded as a list of 2 dim tuples where tuples
        represent intervals
        :param method: for PTTK give value "tt"; otherwise, give "tucker" for parametric Tucker.
        :param tol: The tolerance given to the tensor-train cross approximation method.
        :param ell: The rank given to the HOOI method used to compute the Tucker decomposition of the tensor
        :return: Returns a tuple of the form [storage_cost, S, T] where storage_cost is the storage cost associated
        with PTTK  or parametric Tucker, given in number of floating point numbers, and the numpy matrices S and T are
        the pre-computed offline factors which will later be used in computing the low-rank approximation of the
        parametric kernel matrix during the online phase. Note, if X = Y and method = "tt" then the method
        will return the tuple [storage_cost, Q] where storage_cost is the same as before and Q is the Q matrix defined
        in the global PTTK method which is obtained via QR factorization.
        """
        assert 1E-16 <= tol <= 1E-1
        assert kernel_name in ["thin-plate-spline", "matern", "squared-exponential", "exponential", "multiquadrics",
                               "matern-5/2", "matern-3/2"]
        assert method in ["tucker", "tt"]
        assert type(params) is list
        assert 0 < len(params) < 3
        assert type(params[0]) is tuple
        if kernel_name != "matern":
            assert len(params) < 2
            assert params[0][0] != params[0][1]
        if kernel_name == "matern":
            assert len(params) == 2

        self.param_flag = True
        self.method = method
        psd_flag = True
        self.num_ker_evals = 0
        param_poly_lst = []

        match kernel_name:
            case "matern":
                if params[0][0] == params[0][1]:
                    my_ell = params[0][0]
                    self.curr_kernel_func = lambda diff_sq, nu: KernelApprox._matern_kernel(diff_sq, my_ell, nu)
                    nu_polynomial = InterpolatingPolynomial(params[1][0], params[1][1], self.num_c_points)
                    param_poly_lst = [nu_polynomial]
                elif params[1][0] == params[1][1]:
                    my_nu = params[1][0]
                    self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._matern_kernel(diff_sq, ell, my_nu)
                    ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                    param_poly_lst = [ell_polynomial]
                else:
                    self.curr_kernel_func = lambda diff_sq, ell, nu: KernelApprox._matern_kernel(diff_sq, ell, nu)
                    nu_polynomial = InterpolatingPolynomial(params[1][0], params[1][1], self.num_c_points)
                    ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                    param_poly_lst = [ell_polynomial, nu_polynomial]
            case "squared-exponential":
                self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._squared_exponential(diff_sq, ell)
                ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                param_poly_lst = [ell_polynomial]
            case "exponential":
                self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._exponential(diff_sq, ell)
                ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                param_poly_lst = [ell_polynomial]
            case "multiquadrics":
                self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._multiquadrics(diff_sq, ell)
                ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                param_poly_lst = [ell_polynomial]
                psd_flag = False
            case "matern-5/2":
                my_nu = 5 / 2
                self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._matern_kernel(diff_sq, ell, my_nu)
                ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                param_poly_lst = [ell_polynomial]

            case "matern-3/2":
                my_nu = 3 / 2
                self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._matern_kernel(diff_sq, ell, my_nu)
                ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                param_poly_lst = [ell_polynomial]

            case "thin-plate-spline":
                self.curr_kernel_func = lambda diff_sq, ell: KernelApprox._thin_plate_spline(diff_sq, ell)
                ell_polynomial = InterpolatingPolynomial(params[0][0], params[0][1], self.num_c_points)
                param_poly_lst = [ell_polynomial]
                psd_flag = False

        self.param_c_matrix = np.zeros((self.num_c_points, len(param_poly_lst)))
        for i in range(len(param_poly_lst)):
            poly = param_poly_lst[i]
            self.param_c_matrix[:, i] = np.squeeze(np.array(poly.get_nodes()))
        my_vec_ker = lambda M: self._vec_kernel(M, self.curr_kernel_func, len(param_poly_lst))
        if method == "tt":
            self.ttk_instance = _ParamTensorTrainKernel(self.source_matrix, self.target_matrix, self.source_poly_lst,
                                                        param_poly_lst, self.target_poly_lst, my_vec_ker,
                                                        self.num_c_points)
            if np.array_equal(self.source_matrix, self.target_matrix):
                return (self.ttk_instance.offline_mode(tol, psd_flag=psd_flag, sym_flag=True),
                        self.ttk_instance.get_Q_factor())
            else:
                sto = self.ttk_instance.offline_mode(tol, psd_flag=psd_flag)
                S, T = self.ttk_instance.get_S_and_T()
                return sto, S, T
        elif method == "tucker":
            self.ttk_instance = Param_Kernel_Approx(self.source_matrix, self.target_matrix, self.source_poly_lst,
                                                    self.target_poly_lst, param_poly_lst, self.curr_kernel_func, ell,
                                                    self.num_c_points)
            self.num_ker_evals = (self.num_c_points) ** (len(self.source_poly_lst) + len(self.target_poly_lst) +
                                                         len(param_poly_lst))
            return self.ttk_instance.sto, self.ttk_instance.Fs, self.ttk_instance.Ft

    def online_mode(self, param, compress_flag=True):
        """
        online_mode is a public method that performs the online phase of either PTTK or parametric Tucker depending on
        the value passed to the variable method in the public function parametric_kernel_approximation
        :param param: a tuple or list containing the specific parameters for which a low-rank approximation of the
         parametric kernel matrix must be computed for
        :param compress_flag: if compress_flag = False then perform PTTK-Global-1; otherwise, perform PTTK-Global-2
        :return: If X != Y then the method returns the matrix M such that S*M*transpose(T) gives a low-rank
        approximation of the parametric kernel matrix instantiated by the parameters given by the argument param. If
        X = Y and if compress_flag = False then we return a matrix W such that Q*W*transpose(Q) gives a symmetric
        low-rank approximation of the parametric kernel matrix instantiated by the parameters given by the argument
        param. If X = Y and if compress_flag = True then we return a tuple of matrices [Wh, Uh] such that
        (Q*Uh)*Wh*transpose(Q*Uh) gives a low-rank approximation of the parametric kernel matrix  instantiated by
        the parameters given by the argument param.
        """
        assert self.param_flag is True
        assert type(param) is list or param is tuple
        if self.method == 'tt':
            return self.ttk_instance.online_mode(param, compress_flag)
        else:
            return self.ttk_instance.online_mode(param)

    def generate_true_parametric_kernel_mat(self, param, s_l=None, s_r=None):
        """
        generate_true_parametric_kernel_mat is a public method that generates the parametric kernel matrix (not an
        approximation) instantiated by the kernel specified by the argument kernel_name passed to the public method
        parametric_kernel_approximation and the arguments source_matrix and target_matrix passed
        to the class constructor.
        :param param: a tuple or list containing the specific parameters for which the parametric kernel matrix
        must be computed for
        :param s_l: the rows for which to compute the sub-sampled parametric kernel matrix
        :param s_r: the columns for which to compute the sub-sampled parametric kernel matrix
        :return: Returns the numpy matrix that represents the parametric kernel matrix induced by the class object and
        the value of the argument param if the variables s_l and s_r have default values.
        Otherwise, returns the numpy matrix that represents the sub-sampled parametric kernel matrix induced by the
        class object and the values of the argument param, s_l, and s_r.
        """
        assert self.param_flag is True
        assert type(param) is tuple or type(param) is list
        param = tuple(param)
        if s_l is not None and s_r is not None:
            dist_mat = cdist(self.source_matrix[s_l, :], self.target_matrix[s_r, :])
        else:
            dist_mat = cdist(self.source_matrix, self.target_matrix)
        dist_mat2 = dist_mat ** 2
        kernel_mat = self.curr_kernel_func(dist_mat2, *param)
        return kernel_mat

    def generate_true_kernel_mat(self, s_l=None, s_r=None):
        """
        generate_true_kernel_mat is a public method that generates the  kernel matrix (not an
        approximation) instantiated by the kernel specified by the argument kernel_name passed to the public method
        kernel_approximation
        :param s_l: the rows for which to compute the sub-sampled  kernel matrix
        :param s_r: the columns for which to compute the sub-sampled  kernel matrix
        :return: Returns the numpy matrix that represents the kernel matrix induced by the class object if the variables
        s_l and s_r have default values. Otherwise, returns the numpy matrix that represents the
        sub-sampled kernel matrix induced by the  class object and the values of the argument  s_l and s_r.
        """
        assert self.param_flag is False
        if s_l is not None and s_r is not None:
            dist_mat = cdist(self.source_matrix[s_l, :], self.target_matrix[s_r, :])
        else:
            dist_mat = cdist(self.source_matrix, self.target_matrix)
        dist_mat2 = dist_mat ** 2
        kernel_mat = self.curr_kernel_func(dist_mat2)
        return kernel_mat

    # Where M is a matrix of dimension (num_ker_evals x (2d + param_dim)).
    # We return ker evaluated on the num_ker_eval indices of length (2d + param_dim),
    # where the indices index into the respective Chebyshev points associated with
    # the evaluation tensor.
    def _vec_kernel(self, M, ker, param_dim):
        self.num_ker_evals = self.num_ker_evals + M.shape[0]
        d = self.source_c_matrix.shape[1]
        src = np.zeros((M.shape[0], d))
        trg = np.zeros((M.shape[0], d))

        for i in range(d):
            src[:, i] = self.source_c_matrix[M[:, i], i]
        for i in range(d):
            trg[:, i] = self.target_c_matrix[M[:, i + d + param_dim], i]
        if param_dim > 0:
            par = np.zeros((M.shape[0], param_dim))
            for i in range(param_dim):
                par[:, i] = self.param_c_matrix[M[:, i + d], i]

        if param_dim == 0:
            return ker(np.sum(np.square(src - trg), axis=1))
        elif param_dim == 1:
            return ker(np.sum(np.square(src - trg), axis=1), par)
        elif param_dim == 2:
            return ker(np.sum(np.square(src - trg), axis=1), par[:, 0], par[:, 1])

    def get_curr_kernel_function(self):
        return self.curr_kernel_func

    @staticmethod
    def evaluate_error(kernel_mat_true, kernel_mat_approx, my_ord):
        """
        evaluate_error is a public static method the computes the relative error between the numpy matrices
        kernel_mat_true and kernel_mat_approx in the norm specified by my_ord such that kernel_mat_approx is an
        approximation to kernel_mat_true
        :param kernel_mat_true: the true matrix, numpy matrix
        :param kernel_mat_approx: the approximation matrix, numpy matrix
        :param my_ord: order of the norm used
        :return: relative error of kernel_mat_true and kernel_mat_approx in the norm specified by my_ord
        """
        if my_ord == "max":
            return np.abs(kernel_mat_true - kernel_mat_approx).max() / np.abs(kernel_mat_true).max()
        else:
            return np.linalg.norm(kernel_mat_true - kernel_mat_approx, ord=my_ord) / np.linalg.norm(kernel_mat_true,
                                                                                                    ord=my_ord)

    @staticmethod
    def _exponential(diff_sq, ell):
        return np.exp(-1 * (np.sqrt(np.squeeze(diff_sq)) / np.squeeze(ell)))

    @staticmethod
    def _squared_exponential(diff_sq, ell):
        return np.exp(-1 * (np.squeeze(diff_sq) / np.squeeze(ell) ** 2))

    @staticmethod
    def _matern_kernel(diff_sq, ell=1, nu=1):
        ell = np.squeeze(ell)

        if type(nu) is not type(np.zeros((1, 1))):
            if nu == 5 / 2:
                return ((1 + np.sqrt(5) * np.sqrt(diff_sq) / ell + 5 * diff_sq / (3 * ell * ell)) *
                        np.exp(-1 * np.sqrt(5) * np.sqrt(diff_sq) / ell))
            if nu == 3 / 2:
                return (1 + np.sqrt(3) * np.sqrt(diff_sq) / ell) * np.exp(-1 * np.sqrt(3) * np.sqrt(diff_sq) / ell)

        l = ell
        v = nu
        r = np.abs(np.sqrt(diff_sq))
        part1 = 2 ** (1 - v) / gamma(v)
        part2 = (np.sqrt(2 * v) * r / l) ** v
        part3 = kv(v, np.sqrt(2 * v) * r / l)
        return part1 * part2 * part3

    @staticmethod
    def _bi_harmonic(diff_sq):
        return np.power(diff_sq, -1)

    @staticmethod
    def _laplace2D(diff_sq):
        return -1 * np.log(np.sqrt(diff_sq))

    @staticmethod
    def _laplace3D(diff_sq):
        return np.power(np.sqrt(diff_sq), -1)

    @staticmethod
    def _multiquadrics(diff_sq, ell):
        return np.sqrt((diff_sq / (np.squeeze(ell) ** 2)) + 1)

    @staticmethod
    def _thin_plate_spline(diff_sq, ell):
        part_1 = (diff_sq / np.power(np.squeeze(ell), 2))
        part_2 = np.log(diff_sq / np.power(np.squeeze(ell), 2))
        return part_1 * part_2

    @staticmethod
    def _thin_plate(diff_sq):
        return diff_sq * np.log(np.sqrt(diff_sq))

    @staticmethod
    def _laplace(diff_sq):
        return np.power(np.sqrt(diff_sq), -1)


# public helper classes.
class InterpolatingPolynomial:
    """
    InterpolatingPolynomial is a public class that creates an interpolating polynomial object over the interval [a, b]
    at n Chebyshev nodes of the first kind.
    :param a: the left end point of the interval [a, b]
    :param b: the right end point of the interval [a, b]
    :param n: number of Chebyshev nodes of the first kind taken over the interval [a, b]
    :return: an InterpolatingPolynomial class object
    """
    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n
        self.cheb_nodes = []
        self.arr = np.arange(n - 1)
        self.arr2 = np.arange(n)

        for i in range(n):
            c = (self._I(self._eta_k(i + 1)))
            self.cheb_nodes.append(c)

        self.nodes = np.array(self.cheb_nodes)

    def _T(self, x, k):
        return np.cos(k * np.arccos(x))

    def _eta_k(self, k):
        n = self.n
        return np.cos(np.pi * (2 * k - 1) / (2 * n))

    def _inv_I(self, x):
        return 2.0 * (x - self.a) / (self.b - self.a) - 1.0

    def _I(self, x):
        return (x + 1.0) * (self.b - self.a) / 2.0 + self.a

    def eval(self, x, y):
        n = self.n
        arr = self.arr
        g = lambda k: self._T(self._inv_I(x), k + 1) * self._T(self._inv_I(y), k + 1)
        arr = g(arr)
        my_sum = np.sum(arr)
        my_sum = (2.0 / n) * my_sum
        my_sum = (1.0 / n) + my_sum

        return my_sum

    def v_eval(self, x_arr, y_arr):
        n = self.n
        arr = self.arr
        xv, yv = np.meshgrid(x_arr, y_arr, indexing='ij', copy=False)
        g = lambda x, k, y: self._T(self._inv_I(x), k + 1) * self._T(self._inv_I(y), k + 1)
        my_sum = g(xv, 0, yv)

        for i in range(1, np.size(arr)):
            my_sum = my_sum + g(xv, i, yv)

        my_sum = (2.0 / n) * my_sum
        my_sum = (1.0 / n) + my_sum
        return my_sum


    def node_eval(self, x):
        """
        node_eval is a public method that evaluates the n Lagrange basis polynomials associated with the interval
        [a, b] at point x
        :param x: point in the interval [a, b]
        :return: list containing the evaluations of the n lagrange basis polynomials [l_1(x), l_2(x), ..., l_n(x)]
        """
        lst = []
        for c in self.cheb_nodes:
            lst.append(self.eval(c, x))
        return lst

    # take r points return a matrix of size rxd
    def vec_node_eval(self, arr):
        """
        vec_node_eval is a public method that computes the numpy matrix of size r \times d where r is the length of
        arr such that the i'th row of the numpy matrix is node_eval(self, arr[i]), this computation is done in a
        vectorized manner
        :return: r \times d matrix of Lagrange basis polynomial evaluations
        """
        return self.v_eval(arr, self.nodes)

    def get_nodes(self):
        """
        get_nodes is a public method that returns the Chebyshev nodes defined on the interval [a, b]
        :return: list containing the n Chebyshev nodes
        """
        return self.cheb_nodes


# private helper classes
class _ParamTensorTrainKernel:
    def __init__(self, X, Y, source_poly_lst, param_poly_lst, target_poly_lst, s, num_c_points):
        self.TT_factors = 0
        self.paramcores = 0
        self.X = X
        self.Y = Y
        self.s = s
        self.sym_flag = False
        self.psd_flag = True
        self.tol = 0
        self.u = [num_c_points] * (len(source_poly_lst) * 2 + len(param_poly_lst))
        self.source_poly_lst = source_poly_lst
        self.target_poly_lst = target_poly_lst
        self.poly_lst = source_poly_lst + param_poly_lst + target_poly_lst
        self.param_dim = len(param_poly_lst)
        self.param_poly_lst = param_poly_lst
        self.d = len(self.source_poly_lst)
        self.len = len(source_poly_lst) + len(target_poly_lst) + len(param_poly_lst)
        self.n = num_c_points
        self.U_lst, self.V_lst = self._form_U_and_V()

    def _form_U_and_V(self):
        source_lst = self.source_poly_lst
        target_lst = self.target_poly_lst
        U_lst = []
        V_lst = []

        for i in range(len(source_lst)):
            temp_polynomial = source_lst[i]
            U_lst.append(temp_polynomial.vec_node_eval(self.X[:, i]))

        for i in range(len(target_lst)):
            temp_polynomial = target_lst[i]
            V_lst.append(np.transpose(temp_polynomial.vec_node_eval(self.Y[:, i])))

        return U_lst, V_lst

    # TODO: implement a round method that operates on tt cores.
    def _form_factors(self, ep):
        cores = greedy_cross(self.u, self.s, ep, 1500)
        cores = [torch.tensor(x) for x in cores]
        tt = tn.tensor.Tensor(cores)
        tt.round_tt(ep / 2)
        print("TT-decomposition of evaluation tensor: ")
        print(tt)
        cores = [x.numpy(force=True) for x in tt.cores]
        return cores

    def pre_process(self, my_epp):
        self.TT_factors = self._form_factors(my_epp)
        left_factor, right_factor = self._precompute(self.TT_factors,  self.U_lst, self.V_lst)
        param_cores = self.TT_factors[self.d: len(self.TT_factors) - self.d]
        left_factor = left_factor.reshape((1, np.shape(left_factor)[0], np.shape(left_factor)[1]), order='F')
        right_factor = right_factor.reshape((np.shape(right_factor)[0], np.shape(right_factor)[1], 1), order='F')
        cores = [left_factor] + param_cores + [right_factor]
        tt = tn.tensor.Tensor([torch.tensor(x) for x in cores])
        tt.round_tt(my_epp / 2)
        return [x.numpy(force=True) for x in tt.cores]

    def get_Q_factor(self):
        assert self.paramcores != 0 and self.sym_flag
        return self.paramcores[0][0]

    def get_S_and_T(self):
        assert self.paramcores != 0 and not self.sym_flag
        left_factor = self.paramcores[0]
        right_factor = self.paramcores[-1]
        (s1, s2, s3) = np.shape(right_factor)
        right_factor = np.reshape(right_factor, (s1, s2), order='F')
        (s1, s2, s3) = np.shape(left_factor)
        left_factor = np.reshape(left_factor, (s2, s3), order='F')
        return left_factor, right_factor

    def offline_mode(self, my_epp, psd_flag=True, sym_flag=False):
        self.psd_flag = psd_flag
        self.tol = my_epp
        self.sym_flag = sym_flag

        cores = self.pre_process(my_epp)
        if self.sym_flag:
            Q, R = sci.linalg.qr(np.block([np.squeeze(cores[0]), np.squeeze(cores[-1]).T]),
                                 overwrite_a=True, mode='economic')
            cores[0] = (Q, R)
            cores.pop()
            self.paramcores = cores
        else:
            self.paramcores = cores

        sto_size = 0
        for core in self.paramcores:
            if type(core) is tuple:
                sto_size = sto_size + np.size(core[0]) + np.size(core[1])
            else:
                sto_size = sto_size + np.size(core)

        return sto_size

    def online_mode(self, param, compress_flag=True):
        factor_lst = []
        my_counter = 0

        for i in range(1, 1 + self.param_dim):
            core = self.paramcores[i]
            vec = self.param_poly_lst[my_counter].node_eval(param[my_counter])
            my_counter = my_counter + 1
            core = self.ttm(core, np.array(vec))
            (s1, s2) = np.shape(core)
            core = np.reshape(core, (s1, s2), order='F')
            factor_lst.append(core)

        if len(factor_lst) > 1:
            middle_factor = np.linalg.multi_dot(factor_lst)
        else:
            middle_factor = factor_lst[0]

        if self.sym_flag:
            r_1 = middle_factor.shape[0]
            r_2 = middle_factor.shape[1]
            r = r_1 + r_2
            middle_factor_n = np.zeros([r, r])
            middle_factor_n[:r_1, r_1:r] = middle_factor
            middle_factor_n[r_1:r, :r_1] = np.transpose(middle_factor)
            middle_factor = (1 / 2) * middle_factor_n

            R = self.paramcores[0][1]
            RT = R.T
            Z = R @ middle_factor @ RT
            eigenvalues, eigenvectors = np.linalg.eigh(.5 * (Z + Z.T))
            if self.psd_flag:
                eigenvalues[eigenvalues <= 0] = 0
            # Sort eigenvalues and eigenvectors in descending order
            if compress_flag:
                idx = np.argsort(np.abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                cumulative_sum = np.cumsum(np.square(eigenvalues))
                total_sum = cumulative_sum[-1]
                cumulative_sum = np.abs(np.subtract(total_sum, cumulative_sum))
                cumulative_sum[cumulative_sum >= (self.tol ** 2) * total_sum] = -1 * np.inf
                r = int(min(np.argmax(cumulative_sum) + 2, np.size(eigenvalues) - 1))
                V = eigenvectors[:, :r]
                S = np.diag(eigenvalues)[:r, :r]
            else:
                V = eigenvectors
                S = np.diag(eigenvalues)
            return V, S
        else:
            return middle_factor

    def _precompute(self, my_factors,  U_lst, V_lst):
        tt_factors = my_factors[0:self.d] + my_factors[len(my_factors) - self.d:len(my_factors)]
        z = len(tt_factors)
        mid = int(z / 2)

        for i in range(mid):
            (s1, s2, s3) = np.shape(tt_factors[i])
            tt_factors[i] = np.reshape(tt_factors[i], (s1 * s2, s3), order='F')

        for i in range(mid, z):
            (s1, s2, s3) = np.shape(tt_factors[i])
            tt_factors[i] = np.reshape(tt_factors[i], (s1, s2 * s3), order='F')

        u = U_lst[0] @ tt_factors[0]
        for t in range(1, self.d):
            u = fast_dot_mult_1(U_lst[t], u, tt_factors[t])

        tt_factors = tt_factors[mid:]
        tt_factors = list(tt_factors)[::-1]
        V_lst = V_lst[::-1]

        v = tt_factors[0] @ V_lst[0]
        for t in range(1, self.d):
            v = fast_dot_mult_2(v, V_lst[t], tt_factors[t])

        return u, v

    @staticmethod
    def ttm(tensor, vector):
        M = np.einsum('ijk, j -> ik', tensor, vector, optimize=True)
        return M


class _TensorTrainKernel:
    def __init__(self, X, Y, source_poly_lst, target_poly_lst, s, num_c_points):
        self.TT_factors = 0
        self.paramcores = 0
        self.X = X
        self.Y = Y
        self.s = s
        self.u = [num_c_points] * (len(source_poly_lst) * 2)
        self.source_poly_lst = source_poly_lst
        self.target_poly_lst = target_poly_lst
        self.poly_lst = source_poly_lst + target_poly_lst
        self.d = len(self.source_poly_lst)
        self.len = len(source_poly_lst) + len(target_poly_lst)
        self.n = num_c_points
        self.U_lst, self.V_lst = self._form_U_and_V()

    def _form_U_and_V(self):
        source_lst = self.source_poly_lst
        target_lst = self.target_poly_lst
        U_lst = []
        V_lst = []

        for i in range(len(source_lst)):
            temp_polynomial = source_lst[i]
            U_lst.append(temp_polynomial.vec_node_eval(self.X[:, i]))

        for i in range(len(target_lst)):
            temp_polynomial = target_lst[i]
            V_lst.append(np.transpose(temp_polynomial.vec_node_eval(self.Y[:, i])))

        return U_lst, V_lst

    def _form_factors(self, ep):
        cores = greedy_cross(self.u, self.s, ep, 2000)
        cores = [torch.tensor(x) for x in cores]
        tt = tn.tensor.Tensor(cores)
        tt.round_tt(ep / 2)
        print("TT-decomposition of evaluation tensor: ")
        print(tt)
        cores = [x.numpy(force=True) for x in tt.cores]
        return cores

    def offline_mode(self, my_epp):
        self.TT_factors = self._form_factors(my_epp)
        u, v, A, B = self._precompute(self.TT_factors, self.U_lst, self.V_lst)
        left_factor = u @ A
        right_factor = B @ v
        return left_factor, right_factor

    def _precompute(self, my_factors, U_lst, V_lst):
        tt_factors = my_factors
        z = len(tt_factors)
        mid = int(z / 2)

        for i in range(mid):
            (s1, s2, s3) = np.shape(tt_factors[i])
            tt_factors[i] = np.reshape(tt_factors[i], (s1 * s2, s3), order='F')

        for i in range(mid, z):
            (s1, s2, s3) = np.shape(tt_factors[i])
            tt_factors[i] = np.reshape(tt_factors[i], (s1, s2 * s3), order='F')

        u = U_lst[0]

        for t in range(1, self.d):
            temp1 = U_lst[t]
            temp2 = u @ tt_factors[t - 1]
            u = np.transpose(tl.tenalg.core_tenalg.khatri_rao([np.transpose(temp1), np.transpose(temp2)]))

        tt_factors = tt_factors[mid:z]
        tt_factors = list(tt_factors)[::-1]

        V_lst = V_lst[::-1]
        v = V_lst[0]

        for t in range(0, self.d - 1):
            temp1 = tt_factors[t] @ v
            temp2 = V_lst[t + 1]
            v = tl.tenalg.core_tenalg.khatri_rao([temp1, temp2])

        return u, v, my_factors[self.d - 1], my_factors[self.d]


class Param_Kernel_Approx:

    def __init__(self, X, Y, source_poly_lst, target_poly_lst, param_poly_lst, s, l, num_c_points):
        self.X = X
        self.Y = Y
        self.s = s
        self.l = l
        self.sto = 0
        self.source_poly_lst = source_poly_lst
        self.target_poly_lst = target_poly_lst
        self.param_poly_lst = param_poly_lst
        self.poly_lst = source_poly_lst + target_poly_lst + param_poly_lst
        self.spatial_dim = len(source_poly_lst)
        self.param_dim = len(param_poly_lst)
        self.n = num_c_points
        self.N = self._formTensorN()
        self.G, self.A_lst = tl.decomposition.tucker(tl.tensor(self.N), rank=l)
        self.Fs, self.Ft = self._offline_mode()

    def _formTensorN(self):
        nodes_lst = []
        for p in self.poly_lst:
            nodes_lst.append(p.get_nodes())
        m_points = np.meshgrid(*tuple(nodes_lst), indexing='ij')
        diff_ten = 0
        for i in range(0, self.spatial_dim):
            diff_ten = (m_points[i] - m_points[self.spatial_dim + i]) ** 2 + diff_ten
        return self.s(diff_ten, *m_points[2 * self.spatial_dim: 2 * self.spatial_dim + self.param_dim])

    def _function_transform(self, lst):
        diff = 0
        for i in range(self.spatial_dim):
            diff = lst[i] - lst[i + self.spatial_dim] + diff
        return self.s(diff, )

    def _offline_mode(self):
        source_lst = self.source_poly_lst
        target_lst = self.target_poly_lst
        U_lst = []
        V_lst = []

        for i in range(len(source_lst)):
            temp_polynomial = source_lst[i]
            U = temp_polynomial.vec_node_eval(self.X[:, i]) @ self.A_lst[i]
            U_lst.append(np.transpose(U))

        for i in range(len(target_lst)):
            temp_polynomial = target_lst[i]
            V = temp_polynomial.vec_node_eval(self.Y[:, i]) @ self.A_lst[i + len(source_lst)]
            V_lst.append(np.transpose(V))

        F_s = tl.tenalg.core_tenalg.khatri_rao(U_lst, reverse=True)
        F_t = tl.tenalg.core_tenalg.khatri_rao(V_lst, reverse=True)
        F_s = np.transpose(F_s)
        F_t = np.transpose(F_t)

        self.sto = np.size(F_s) + np.size(F_t) + np.size(self.G)

        return F_s, F_t

    def _online_mode(self, theta):

        s_lst = []

        for i in range(len(theta)):
            s_temp = self.param_poly_lst[i]
            eval_temp = np.zeros((1, self.n))
            eval_temp[0, :] = np.asarray(s_temp.node_eval(theta[i]))
            A_temp = self.A_lst[len(self.source_poly_lst) + len(self.target_poly_lst) + i]
            s_lst.append(eval_temp @ A_temp)

        temp_modes = np.arange(len(self.source_poly_lst) + len(self.target_poly_lst), len(self.poly_lst))
        T_theta = tl.tenalg.core_tenalg.multi_mode_dot(self.G, s_lst, modes=temp_modes)

        D = len(self.source_poly_lst)
        temp_size = int(np.power(self.l, D))
        M_theta = np.reshape(T_theta, (temp_size, temp_size), order="F")

        return M_theta

    def online_mode(self, theta):
        M_theta = self._online_mode(theta)
        return M_theta


class Tucker_Kernel_Approx:

    def __init__(self, X, Y, source_poly_lst, target_poly_lst, s, l, num_c_points):
        self.X = X
        self.Y = Y
        self.s = s
        self.l = l
        self.source_poly_lst = source_poly_lst
        self.target_poly_lst = target_poly_lst
        self.poly_lst = source_poly_lst + target_poly_lst
        self.spatial_dim = len(source_poly_lst)
        self.n = num_c_points
        self.N = self._formTensorN()
        self.G, self.A_lst = tl.decomposition.tucker(tl.tensor(self.N), l)
        self.Fs, self.Ft = self._offline_mode()

    def _formTensorN(self):
        nodes_lst = []
        for p in self.poly_lst:
            nodes_lst.append(p.get_nodes())
        m_points = np.meshgrid(*tuple(nodes_lst), indexing='ij')
        diff_ten = 0
        for i in range(0, self.spatial_dim):
            diff_ten = (m_points[i] - m_points[self.spatial_dim + i]) ** 2 + diff_ten
        return self.s(diff_ten)

    def _offline_mode(self):
        source_lst = self.source_poly_lst
        target_lst = self.target_poly_lst
        U_lst = []
        V_lst = []

        for i in range(len(source_lst)):
            temp_polynomial = source_lst[i]
            U = temp_polynomial.vec_node_eval(self.X[:, i]) @ self.A_lst[i]
            U_lst.append(np.transpose(U))

        for i in range(len(target_lst)):
            temp_polynomial = target_lst[i]
            V = temp_polynomial.vec_node_eval(self.Y[:, i]) @ self.A_lst[i + len(source_lst)]
            V_lst.append(np.transpose(V))

        F_s = tl.tenalg.core_tenalg.khatri_rao(U_lst, reverse=True)
        F_t = tl.tenalg.core_tenalg.khatri_rao(V_lst, reverse=True)
        F_s = np.transpose(F_s)
        F_t = np.transpose(F_t)
        return F_s, F_t

    def offline_mode(self):
        temp_size = int(np.power(self.l, self.spatial_dim))
        G = np.reshape(self.G, (temp_size, temp_size), order='F')
        return self.Fs, G, np.transpose(self.Ft)


@jit(nopython=True)
def fast_dot_mult_1(A, B, C):
    R = np.zeros((A.shape[0], C.shape[1]))
    for i in range(A.shape[0]):
        R[i, :] = np.outer(A[i, :], B[i, :]).reshape(1, A.shape[1] * B.shape[1]) @ C
    return R


@jit(nopython=True)
def fast_dot_mult_2(A, B, C):
    R = np.zeros((C.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        R[:, i] = C @ np.outer(A[:, i], B[:, i]).reshape(A.shape[0] * B.shape[0])
    return R
