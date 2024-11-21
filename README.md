# Special Unitary Parameterized Estimators of Rotation (SUPER)
[![arXiv](https://img.shields.io/badge/arXiv-2411.13109-b31b1b.svg)](https://arxiv.org/abs/2411.13109)

💥Code for paper [Special Unitary Parameterized Estimators of Rotation](https://arxiv.org/abs/2411.13109) 🌐🌀🔢🧠

## Rotation Estimation from Point Correspondences
C++ code for classic Wahba's problem in `wahba` folder. `SUPER.hpp` includes the following algorithms:
-  General solution for stereographic point inputs (G_P)
-  General solution for 3D inputs (G_S)
-  Efficient 2 point algorithms (weighted and unweighted) for 3D points. Unweighted is particularly efficient.
-  Robust 1pt and 2pt noiseless algorithms for aligning 3D points. 1pt rotations return quaternions with an element as 0 allowing for faster point rotation and composition of rotations.
-  Möbius transformation approximate solutions (G_M) for stereographic point inputs. For 3 points, uses closed-form solution.

For 3D input algorithms, points are assumed to have unit norm. For general solutions involving eigendecomposition (solver option EIGVEC or MOBIUS), compiler flag JACOBI_PD must be defined.

Thanks to [jacobi_pd](https://github.com/jewettaij/jacobi_pd) for providing symmetric matrix diagonalization code!

## Representations for Learning Rotations in Neural Networks
PyTorch implementations of proposed rotation representations for learning (**2-vec** and **QuadMobius**) in `learning` folder. They are easy to drag and drop into any model pipeline to map neural network outputs to a rotation:
```
import torch
from learning.SUPER_maps import *

batch_size = 128

#2-vec
input_data1 = torch.randn(batch_size, 6, dtype=torch.float32) # N x 6 real data
R = map_2vec(input_data1) # N x 3 x 3 tensor of rotation matrices

# whether to use algebraic method or SVD method for backward/forward passes in QuadMobius approaches
alg_bkwd, alg_fwd = False, True

#QuadMobius Real
input_data2 = torch.randn(batch_size, 16, dtype=torch.float32) # N x 16 real data
q = QuadMobiusRotationSolver.apply(input_data2, alg_bkwd, alg_fwd) # N x 4 tensor of quaternions (w, x, y, z)

#QuadMobius Complex
input_data3 = torch.randn(batch_size, 10, dtype=torch.complex64) # N x 10 complex data
q = QuadMobiusRotationSolver.apply(input_data3, alg_bkwd, alg_fwd) # N x 4 tensor of quaternions (w, x, y, z)
```
