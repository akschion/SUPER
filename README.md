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
