# Special Unitary Parameterized Estimators of Rotation (SUPER)
💥Code for paper Special Unitary Parameterized Estimators of Rotation 🌐🌀🔢🧠

## Rotation Estimation from Point Correspondences
C++ code for classic Wahba's problem in `wahba` folder. `SUPER.hpp` includes the following algorithms:
-  general solution for stereographic point inputs (G_P)
-  general solution for 3D inputs (G_S)
-  2 point general algorithm (weighted and unweighted) for 3D points
-  1pt and 2pt noiseless algorithms for aligning 3D points
-  Möbius transformation approximate solutions (G_M) for stereographic point inputs

For 3D input algorithms, points are assumed to have unit norm. For general solutions involving eigendecomposition (solver option EIGVEC or MOBIUS), compiler flag JACOBI_PD must be defined.

Thanks to [jacobi_pd](https://github.com/jewettaij/jacobi_pd) for providing symmetric matrix diagonalization code!

## Representations for Learning Rotations in Neural Networks
The folder `learning` has PyTorch implementations of rotation representations for learning. They are easy to drag and drop into any project to map neural network outputs into a rotation:
```
import torch
from learning.SUPER_maps import *

batch_size = 128
alg_bkwd, alg_fwd = False, True #whether to use algebraic method or SVD method for backward/forward passes

#2-vec
input_data1 = torch.randn(batch_size, 6, dtype=torch.float32) # N x 6 real data
R = map_2vec(input_data1) # N x 3 x 3 tensor of rotation matrices

#QuadMobius Real
input_data2 = torch.randn(batch_size, 16, dtype=torch.float32) # N x 16 real data
q = QuadMobiusRotationSolver.apply(input_data2, alg_bkwd, alg_fwd) # N x 4 tensor of quaternions (w, x, y, z)

#QuadMobius Complex
input_data3 = torch.randn(batch_size, 10, dtype=torch.complex64) # N x 10 complex data
q = QuadMobiusRotationSolver.apply(input_data3, alg_bkwd, alg_fwd) # N x 4 tensor of quaternions (w, x, y, z)
```
