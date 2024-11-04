# ParametricTensorTrainKernel

# Parametric kernel low-rank approximations using tensor train decomposition
This repository contains Python code for computing low-rank approximations of kernel matrices with the tensor train decomposition. It accompanies the paper
> Khan, A., & Saibaba, A. K. (2024). Parametric kernel low-rank approximations using tensor train decomposition. Submitted. [arXiv preprint](https://arxiv.org/abs/2406.06344).

## Requirements
The [Python](python/) code requires the following packages for test problems.
1. [numpy](https://github.com/numpy/numpy)
2. [tensorly](https://github.com/scipy/scipy)
3. [pytorch](https://github.com/pytorch/pytorch)
4. [numba](https://github.com/numba/numba)
5. [scikit-learn](https://github.com/scikit-learn/scikit-learn)
6. [tntorch(forked)](https://github.com/awkhan3/tntorch)
7. [py-markdown-table](https://pypi.org/project/py-markdown-table/)

## License
To use these codes in your research, please see the [License](LICENSE). If you find our code useful, please consider citing our paper.
```bibtex
@article{khan2024parametric,
  title={Parametric kernel low-rank approximations using tensor train decomposition},
  author={Khan, Abraham and Saibaba, Arvind K},
  journal={arXiv preprint arXiv:2406.06344},
  year={2024}
}
```
## Python Enviroment Setup
1. `chmod +x setup_env.sh`
2. `./setup_env`

## Generating Tables
1. `chmod +x run_experiments.sh`
2. `./run_experiments.sh`

## Funding
This work was funded by the National Science Foundation through the awards DMS-1845406, DMS-1821149, and 
DMS-2026830.


 
