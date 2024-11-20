/*************************************************
BSD 3-Clause License

Copyright (c) 2024, Akshay Chandrasekhar

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
************************************************/

#ifndef SUPER_HPP
#define SUPER_HPP

#define _USE_MATH_DEFINES
#include <array>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <complex>

#if JACOBI_PD
//open source library for matrix diagonalization, used as reference solution to eigenvector problem
#include "jacobi_pd.hpp"
#include "jacobi_hermitian.hpp"
#endif
#if EIGEN
//reference for complex diagonalization
#include <Eigen/Eigenvalues>
#endif

/**
 * Algorithms from "Special Unitary Parameterized Estimators of Rotation" by Akshay Chandrasekhar
 */

///////////////Aligning rotations (noiseless one and two point estimation)

//Finds quaternion that aligns two points (one point noiseless case)
//Defaults to 180 rotation if stable, otherwise, chooses another rotation with an element of the quaternion equal to 0
template<class T>
void alignQuatSUPER(std::array<T, 3> &refpt, std::array<T, 3> &tarpt, std::array<T, 4> &q_align) {
    constexpr T tol = std::is_same<T, double>::value ? 1e-5 : 1e-3f; //tolerance for using rotation
    const T &x = refpt[0], &y = refpt[1], &z = refpt[2], &m = tarpt[0], &n = tarpt[1], &p = tarpt[2];

    const T xpm = x + m, ypn = y + n, zpp = z + p;
    if (fabs(xpm) > tol || fabs(ypn) > tol || fabs(zpp) > tol) {
        //180 rotation (reflection)
        const T norm = 1. / sqrt(fma(xpm, xpm, fma(ypn, ypn, zpp * zpp)));
        q_align[0] = T(0), q_align[1] = xpm * norm, q_align[2] = ypn * norm, q_align[3] = zpp * norm;
    } else {
        //near antiparallel
        //choose rotation with one of the axis elements 0
        const T mmx = m - x, ymn = y - n;
        const T zpp2 = zpp * zpp, mmx2 = mmx * mmx, ymn2 = ymn * ymn;

        if (zpp2 > tol || mmx2 > tol) {
            //z_q = 0
            T norm = T(1) / sqrt(zpp2 + mmx2 + ymn2);
            q_align[0] = zpp * norm, q_align[1] = ymn * norm, q_align[2] = mmx * norm, q_align[3] = T(0);
        } else {
            const T zmp = z - p;
            const T zmp2 = zmp * zmp, xpm2 = xpm * xpm;
            if (xpm2 > tol || ymn2 > tol) {
                //x_q = 0
                const T norm = T(1) / sqrt(xpm2 + zmp2 + ymn2);
                q_align[0] = xpm * norm, q_align[1] = T(0), q_align[2] = zmp * norm, q_align[3] = -ymn * norm;
            } else {
                //y_q = 0
                const T norm = T(1) / sqrt(fma(ypn, ypn, zmp2 + mmx2));
                q_align[0] = ypn * norm, q_align[1] = -zmp * norm, q_align[2] = T(0), q_align[3] = -mmx * norm;
            }
        }
    }
}

//Rotate vectors by which element is 0. Assumes unit q = (w, x, y, z), w is scalar
// x = 0
template<class T>
void rotateVecByQuatX0(std::array<T, 4> &q, std::array<T, 3> &v) {
    T t1 = 2 * fma(q[2], v[2], -q[3] * v[1]);
    T t2 = 2 * q[3] * v[0];
    T t3 = -2 * q[2] * v[0];
    v[0] += fma(q[0], t1, fma(q[2], t3, -q[3] * t2));
    v[1] += fma(q[0], t2, q[3] * t1);
    v[2] += fma(q[0], t3, -q[2] * t1);
}

// y = 0
template<class T>
void rotateVecByQuatY0(std::array<T, 4> &q, std::array<T, 3> &v) {
    T t1 = -2 * q[3] * v[1];
    T t2 = 2 * fma(q[3], v[0], -q[1] * v[2]);
    T t3 = 2 * q[1] * v[1];
    v[0] += fma(q[0], t1, -q[3] * t2);
    v[1] += fma(q[0], t2, fma(q[3], t1, -q[1] * t3));
    v[2] += fma(q[0], t3, q[1] * t2);
}

// z = 0
template<class T>
void rotateVecByQuatZ0(std::array<T, 4> &q, std::array<T, 3> &v) {
    T t1 = 2 * q[2] * v[2];
    T t2 = -2 * q[1] * v[2];
    T t3 = 2 * fma(q[1], v[1], -q[2] * v[0]);
    v[0] += fma(q[0], t1, q[2] * t3);
    v[1] += fma(q[0], t2, -q[1] * t3);
    v[2] += fma(q[0], t3, fma(q[1], t2, -q[2] * t1));
}

// w = 0
//180 degree rotations, can be performed easily as reflections
template<class T>
void rotateVecByQuatW0(std::array<T, 4> &q, std::array<T, 3> &v) {
    T dot = 2 * fma(v[0], q[1], fma(v[1], q[2], v[2] * q[3]));
    v[0] = fma(dot, q[1], -v[0]);
    v[1] = fma(dot, q[2], -v[1]);
    v[2] = fma(dot, q[3], -v[2]);
}

//2 point noiseless solution from kernels of Q1 and Q2
//Solution implemented for each of the 36 cases. depending on which case is most stable (max_combo)
//Chooses kernel of one Q constraint and tests both rows of other Q constraint for first solution yielded
//If no valid row, then the reference and target points are roughly collinear
template <class T>
void alignQuatSUPER_2PT(std::array<T, 3> &refpt1, std::array<T, 3> &refpt2, std::array<T, 3> &tarpt1,
                        std::array<T, 3> &tarpt2, std::array<T, 4> &q_align, bool normalize= true) {
    constexpr T tol = std::is_same<T, double>::value ? 1e-10 : 1e-6f; //tolerance for Q2 constraint selection
    T x1 = refpt1[0], y1 = refpt1[1], z1 = refpt1[2], x2 = refpt2[0], y2 = refpt2[1], z2 = refpt2[2];
    const T &m1 = tarpt1[0], &n1 = tarpt1[1], &p1 = tarpt1[2], &m2 = tarpt2[0], &n2 = tarpt2[1], &p2 = tarpt2[2];

    T xpm1 = x1 + m1, ypn1 = y1 + n1, zpp1 = z1 + p1, xmm1 = x1 - m1, ymn1 = y1 - n1, zmp1 = z1 - p1;
    T xpm2 = x2 + m2, ypn2 = y2 + n2, zpp2 = z2 + p2, xmm2 = x2 - m2, ymn2 = y2 - n2, zmp2 = z2 - p2;

    //find maximum element among (ref - tar) and (ref + tar) for each point
    T max_val1 = fabs(xpm1), max_val2 = fabs(xpm2);
    int arg1 = 0, arg2 = 0;
    if (fabs(ypn1) > max_val1) {max_val1 = fabs(ypn1); arg1 = 1;}
    if (fabs(zpp1) > max_val1) {max_val1 = fabs(zpp1); arg1 = 2;}
    if (fabs(xmm1) > max_val1) {max_val1 = fabs(xmm1); arg1 = 3;}
    if (fabs(ymn1) > max_val1) {max_val1 = fabs(ymn1); arg1 = 4;}
    if (fabs(zmp1) > max_val1) {max_val1 = fabs(zmp1); arg1 = 5;}
    if (fabs(ypn2) > max_val2) {max_val2 = fabs(ypn2); arg2 = 1;}
    if (fabs(zpp2) > max_val2) {max_val2 = fabs(zpp2); arg2 = 2;}
    if (fabs(xmm2) > max_val2) {max_val2 = fabs(xmm2); arg2 = 3;}
    if (fabs(ymn2) > max_val2) {max_val2 = fabs(ymn2); arg2 = 4;}
    if (fabs(zmp2) > max_val2) {max_val2 = fabs(zmp2); arg2 = 5;}

    //find first element among (ref - tar) and (ref + tar) that is greater than tol for each point
//    int arg1 = fabs(xpm1) > tol ? 0 : (fabs(ypn1) > tol ? 1 : (fabs(zpp1) > tol ? 2 : (fabs(xmm1) > tol ? 3 : (fabs(ymn1) > tol ? 4 : 5))));
//    int arg2 = fabs(xpm2) > tol ? 0 : (fabs(ypn2) > tol ? 1 : (fabs(zpp2) > tol ? 2 : (fabs(xmm2) > tol ? 3 : (fabs(ymn2) > tol ? 4 : 5))));

    int max_combo = (arg1 << 3) + arg2; //encodes with bit shift, avoiding multiplication
    T a, b;
    T q0, q1, q2, q3;

    //Labels:
    //ref = (x, y, z), tar = (m, n, p)
    //middle character - p = +, m = -
    //last number refers to point index (1 or 2)
    // e.g. zpp2 = z2 + p2

    //36 cases
    switch (max_combo) {
        case 0: //xpm1, xpm2
            //Q2 - fourth row
            a = fma(xpm1, zmp2, -xpm2 * zmp1);
            b = fma(-xpm1, ypn2, xpm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn1, zmp2, -ypn2 * zmp1);
                q3 = fma(ymn1, ypn2, fma(zpp1, zmp2, xmm1 * xpm2));
            } else {
                //Q2 - third row
                a = fma(xpm1, ymn2, -xpm2 * ymn1);
                b = fma(xpm1, zpp2, -xpm2 * zpp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(ypn1, ymn2, fma(zmp1, zpp2, xmm1 * xpm2));
                    q3 = fma(zpp2, -ymn1, zpp1 * ymn2);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 1: //xpm1, ypn2
            //Q2 - fourth row
            a = fma(xpm1, zmp2, -xpm2 * zmp1);
            b = fma(-xpm1, ypn2, xpm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn1, zmp2, -ypn2 * zmp1);
                q3 = fma(ymn1, ypn2, fma(zpp1, zmp2, xmm1 * xpm2));
            } else {
                //Q2 - second row
                a = fma(xpm1, xmm2, fma(ymn1, ypn2, zmp1 * zpp2));
                b = fma(ypn1, -zpp2, ypn2 * zpp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm2, ypn1, -ypn2 * xmm1);
                    q3 = fma(xmm2, zpp1, -zpp2 * xmm1);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 2: //xpm1, zpp2
            //Q2 - third row
            a = fma(xpm1, ymn2, -xpm2 * ymn1);
            b = fma(xpm1, zpp2, -xpm2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn1, ymn2, fma(zmp1, zpp2, xmm1 * xpm2));
                q3 = fma(zpp2, -ymn1, zpp1 * ymn2);
            } else{
                //Q2 - second row
                a = fma(xpm1, xmm2, fma(ymn1, ypn2, zmp1 * zpp2));
                b = fma(ypn1, -zpp2, ypn2 * zpp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm2, ypn1, -ypn2 * xmm1);
                    q3 = fma(xmm2, zpp1, -zpp2 * xmm1);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 3: //xpm1, xmm2
            //Q2 - second row
            a = fma(xpm1, xmm2, fma(ymn1, ypn2, zmp1 * zpp2));
            b = fma(ypn1, -zpp2, ypn2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(xmm2, ypn1, -ypn2 * xmm1);
                q3 = fma(xmm2, zpp1, -zpp2 * xmm1);
            } else {
                //Q2 - first row
                a = fma(ymn1, zmp2, -ymn2 * zmp1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm2, zmp1, -xmm1 * zmp2);
                    q3 = fma(xmm2, -ymn1, xmm1 * ymn2);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 4: //xpm1, ymn2
            //Q2 - third row
            a = fma(xpm1, ymn2, -xpm2 * ymn1);
            b = fma(xpm1, zpp2, -xpm2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn1, ymn2, fma(zmp1, zpp2, xmm1 * xpm2));
                q3 = fma(zpp2, -ymn1, zpp1 * ymn2);
            } else {
                //Q2 - first row
                a = fma(ymn1, zmp2, -ymn2 * zmp1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm2, zmp1, -xmm1 * zmp2);
                    q3 = fma(xmm2, -ymn1, xmm1 * ymn2);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 5: //xpm1, zmp2
            //Q2 - fourth row
            a = fma(xpm1, zmp2, -xpm2 * zmp1);
            b = fma(-xpm1, ypn2, xpm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn1, zmp2, -ypn2 * zmp1);
                q3 = fma(ymn1, ypn2, fma(zpp1, zmp2, xmm1 * xpm2));
            } else {
                //Q2 - first row
                a = fma(ymn1, zmp2, -ymn2 * zmp1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm2, zmp1, -xmm1 * zmp2);
                    q3 = fma(xmm2, -ymn1, xmm1 * ymn2);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 8: //ypn1, xpm2
            //Q1 - fourth row
            a = fma(xpm2, zmp1, -xpm1 * zmp2);
            b = fma(-xpm2, ypn1, xpm1 * ypn2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn2, zmp1, -ypn1 * zmp2);
                q3 = fma(ymn2, ypn1, fma(zpp2, zmp1, xmm2 * xpm1));
            } else {
                //Q1 - second row
                a = fma(xpm2, xmm1, fma(ymn2, ypn1, zmp2 * zpp1));
                b = fma(ypn2, -zpp1, ypn1 * zpp2);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm1, ypn2, -ypn1 * xmm2);
                    q3 = fma(xmm1, zpp2, -zpp1 * xmm2);
                } else //Q2k - first row
                    q0 = T(0), q1 = xpm2, q1 = ypn2, q3 = zpp2;
            }
            break;
        case 9: //ypn1, ypn2
            //Q2 - fourth row
            a = fma(ypn1, zmp2, -ypn2 * zmp1);
            b = fma(xpm1, -ypn2, xpm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm1, zmp2, -xpm2 * zmp1);
                q3 = fma(xpm2, xmm1, fma(zpp1, zmp2, ymn1 * ypn2));
            } else {
                //Q2 - second row
                a = fma(-xmm1, ypn2, xmm2 * ypn1);
                b = fma(-zpp2, ypn1, zpp1 * ypn2);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(xpm1, xmm2, fma(zmp1, zpp2, ymn1 * ypn2));
                    q3 = fma(-xmm1, zpp2, xmm2 * zpp1);
                } else //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 10: //ypn1, zpp2
            //Q2 - second row
            a = fma(-xmm1, ypn2, xmm2 * ypn1);
            b = fma(-zpp2, ypn1, zpp1 * ypn2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm1, xmm2, fma(zmp1, zpp2, ymn1 * ypn2));
                q3 = fma(-xmm1, zpp2, xmm2 * zpp1);
            } else {
                //Q2 - third row
                a = fma(xmm1, xpm2, fma(ypn1, ymn2, zmp1 * zpp2));
                b = fma(xpm1, zpp2, -xpm2 * zpp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(xpm1, ymn2, -ymn1 * xpm2);
                    q3 = fma(zpp1, ymn2, -ymn1 * zpp2);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 11: //ypn1, xmm2
            //Q2 - second row
            a = fma(-xmm1, ypn2, xmm2 * ypn1);
            b = fma(-zpp2, ypn1, zpp1 * ypn2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm1, xmm2, fma(zmp1, zpp2, ymn1 * ypn2));
                q3 = fma(-xmm1, zpp2, xmm2 * zpp1);
            } else {
                //Q2 - first row
                a = fma(-xmm1, zmp2, xmm2 * zmp1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(-zmp1, ymn2, ymn1 * zmp2);
                    q3 = fma(xmm1, ymn2, -ymn1 * xmm2);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 12: //ypn1, ymn2
            //Q2 - third row
            a = fma(xmm1, xpm2, fma(ypn1, ymn2, zmp1 * zpp2));
            b = fma(xpm1, zpp2, -xpm2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm1, ymn2, -ymn1 * xpm2);
                q3 = fma(zpp1, ymn2, -ymn1 * zpp2);
            } else {
                //Q2 - first row
                a = fma(-xmm1, zmp2, xmm2 * zmp1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(-zmp1, ymn2, ymn1 * zmp2);
                    q3 = fma(xmm1, ymn2, -ymn1 * xmm2);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 13: //ypn1, zmp2
            //Q2 - fourth row
            a = fma(ypn1, zmp2, -ypn2 * zmp1);
            b = fma(xpm1, -ypn2, xpm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm1, zmp2, -xpm2 * zmp1);
                q3 = fma(xpm2, xmm1, fma(zpp1, zmp2, ymn1 * ypn2));
            } else {
                //Q2 - first row
                a = fma(-xmm1, zmp2, xmm2 * zmp1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(-zmp1, ymn2, ymn1 * zmp2);
                    q3 = fma(xmm1, ymn2, -ymn1 * xmm2);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q1 = ypn1, q3 = zpp1;
            }
            break;
        case 16: //zpp1, xpm2
            //Q1 - third row
            a = fma(xpm2, ymn1, -xpm1 * ymn2);
            b = fma(xpm2, zpp1, -xpm1 * zpp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn2, ymn1, fma(zmp2, zpp1, xmm2 * xpm1));
                q3 = fma(zpp1, -ymn2, zpp2 * ymn1);
            } else {
                //Q1 - second row
                a = fma(xpm2, xmm1, fma(ymn2, ypn1, zmp2 * zpp1));
                b = fma(ypn2, -zpp1, ypn1 * zpp2);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm1, ypn2, -ypn1 * xmm2);
                    q3 = fma(xmm1, zpp2, -zpp1 * xmm2);
                } else //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 17: //zpp1, ypn2
            //Q1 - second row
            a = fma(-xmm2, ypn1, xmm1 * ypn2);
            b = fma(-zpp1, ypn2, zpp2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm2, xmm1, fma(zmp2, zpp1, ymn2 * ypn1));
                q3 = fma(-xmm2, zpp1, xmm1 * zpp2);
            } else {
                //Q1 - third row
                a = fma(xmm2, xpm1, fma(ypn2, ymn1, zmp2 * zpp1));
                b = fma(xpm2, zpp1, -xpm1 * zpp2);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(xpm2, ymn1, -ymn2 * xpm1);
                    q3 = fma(zpp2, ymn1, -ymn2 * zpp1);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 18: //zpp1, zpp2
            //Q2 - second row
            a = fma(-xmm1, zpp2, xmm2 * zpp1);
            b = fma(-ypn1, zpp2, ypn2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm1, xmm2, fma(ymn1, ypn2, zmp1 * zpp2));
                q2 = fma(-xmm1, ypn2, xmm2 * ypn1);
            } else {
                //Q2 - third row
                a = fma(-ymn1, zpp2, ymn2 * zpp1);
                b = fma(xpm1, zpp2, -xpm2 * zpp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(xpm1, ymn2, -xpm2 * ymn1);
                    q2 = fma(xmm1, xpm2, fma(ypn1, ymn2, zmp1 * zpp2));
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q2 = ypn1, q3 = zpp1;
            }
            break;
        case 19: //zpp1, xmm2
            //Q2 - second row
            a = fma(-xmm1, zpp2, xmm2 * zpp1);
            b = fma(-ypn1, zpp2, ypn2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm1, xmm2, fma(ymn1, ypn2, zmp1 * zpp2));
                q2 = fma(-xmm1, ypn2, xmm2 * ypn1);
            } else {
                //Q2 - first row
                a = fma(xmm1, ymn2, -xmm2 * ymn1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(ymn1, zmp2, -zmp1 * ymn2);
                    q2 = fma(-xmm1, zmp2, xmm2 * zmp1);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q2 = ypn1, q3 = zpp1;
            }
            break;
        case 20: //zpp1, ymn2
            //Q2 - third row
            a = fma(-ymn1, zpp2, ymn2 * zpp1);
            b = fma(xpm1, zpp2, -xpm2 * zpp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm1, ymn2, -xpm2 * ymn1);
                q2 = fma(xmm1, xpm2, fma(ypn1, ymn2, zmp1 * zpp2));
            } else {
                //Q2 - first row
                a = fma(xmm1, ymn2, -xmm2 * ymn1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(ymn1, zmp2, -zmp1 * ymn2);
                    q2 = fma(-xmm1, zmp2, xmm2 * zmp1);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q2 = ypn1, q3 = zpp1;
            }
            break;
        case 21: //zpp1, zmp1
            //Q2 - fourth row
            a = fma(xmm1, xpm2, fma(ymn1, ypn2, zpp1 * zmp2));
            b = fma(xpm1, -ypn2, xpm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm1, zmp2, -zmp1 * xpm2);
                q2 = fma(ypn1, zmp2, -zmp1 * ypn2);
            } else {
                //Q2 - first row
                a = fma(xmm1, ymn2, -xmm2 * ymn1);
                b = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(ymn1, zmp2, -zmp1 * ymn2);
                    q2 = fma(-xmm1, zmp2, xmm2 * zmp1);
                } else
                    //Q1k - first row
                    q0 = T(0), q1 = xpm1, q2 = ypn1, q3 = zpp1;
            }
            break;
        case 24: //xmm1, xpm2
            //Q1 - second row
            a = fma(xpm2, xmm1, fma(ymn2, ypn1, zmp2 * zpp1));
            b = fma(ypn2, -zpp1, ypn1 * zpp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(xmm1, ypn2, -ypn1 * xmm2);
                q3 = fma(xmm1, zpp2, -zpp1 * xmm2);
            } else {
                //Q1 - first row
                a = fma(ymn2, zmp1, -ymn1 * zmp2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm1, zmp2, -xmm2 * zmp1);
                    q3 = fma(xmm1, -ymn2, xmm2 * ymn1);
                } else //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 25: //xmm1, ypn2
            //Q1 - second row
            a = fma(-xmm2, ypn1, xmm1 * ypn2);
            b = fma(-zpp1, ypn2, zpp2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm2, xmm1, fma(zmp2, zpp1, ymn2 * ypn1));
                q3 = fma(-xmm2, zpp1, xmm1 * zpp2);
            } else {
                //Q1 - first row
                a = fma(-xmm2, zmp1, xmm1 * zmp2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(-zmp2, ymn1, ymn2 * zmp1);
                    q3 = fma(xmm2, ymn1, -ymn2 * xmm1);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 26: //xmm1, zpp2
            //Q1 - second row
            a = fma(-xmm2, zpp1, xmm1 * zpp2);
            b = fma(-ypn2, zpp1, ypn1 * zpp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm2, xmm1, fma(ymn2, ypn1, zmp2 * zpp1));
                q2 = fma(-xmm2, ypn1, xmm1 * ypn2);
            } else {
                //Q1 - first row
                a = fma(xmm2, ymn1, -xmm1 * ymn2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(ymn2, zmp1, -zmp2 * ymn1);
                    q2 = fma(-xmm2, zmp1, xmm1 * zmp2);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 27: //xmm1, xmm2
            //Q2 - second row
            a = fma(-xmm1, zpp2, xmm2 * zpp1);
            b = fma(xmm1, ypn2, -xmm2 * ypn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q2 = -b, q3 = a;
                q0 = fma(ypn1, -zpp2, zpp1 * ypn2);
                q1 = fma(zmp1, zpp2, fma(ymn1, ypn2, xpm1 * xmm2));
            } else {
                //Q2 - first row
                a = fma(xmm1, ymn2, -xmm2 * ymn1);
                b = fma(xmm1, zmp2, -xmm2 * zmp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q2 = -b, q3 = a;
                    q0 = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                    q1 = fma(ymn1, zmp2, -ymn2 * zmp1);
                } else
                    //Q1k - third row
                    q0 = ypn1, q1 = -zmp1, q2 = T(0), q3 = xmm1;
            }
            break;
        case 28: //xmm1, ymn2
            //Q2 - first row
            a = fma(xmm1, ymn2, -xmm2 * ymn1);
            b = fma(xmm1, zmp2, -xmm2 * zmp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q2 = -b, q3 = a;
                q0 = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                q1 = fma(ymn1, zmp2, -ymn2 * zmp1);
            } else {
                //Q2 - third row
                a = fma(ymn1, -zpp2, ymn2 * zpp1);
                b = fma(xmm1, xpm2, fma(ypn1, ymn2, zmp1 * zpp2)); //drop negative since negating below
                if (fabs(a) > tol || fabs(b) > tol) {
                    q2 = b, q3 = a;
                    q0 = fma(-zpp1, xpm2, xpm1 * zpp2);
                    q1 = fma(-ymn1, xpm2, xpm1 * ymn2);
                } else
                    //Q1k - third row
                    q0 = ypn1, q1 = -zmp1, q2 = T(0), q3 = xmm1;
            }
            break;
        case 29: //xmm1, zmp2
            //Q2 - first row
            a = fma(xmm1, ymn2, -xmm2 * ymn1);
            b = fma(xmm1, zmp2, -xmm2 * zmp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q2 = -b, q3 = a;
                q0 = fma(xpm1, xmm2, fma(ypn1, ymn2, zpp1 * zmp2));
                q1 = fma(ymn1, zmp2, -ymn2 * zmp1);
            } else {
                //Q2 - fourth row
                a = fma(xmm1, xpm2, fma(ymn1, ypn2, zpp1 * zmp2));
                b = fma(-ypn1, zmp2, ypn2 * zmp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q2 = -b, q3 = a;
                    q0 = fma(ypn1, xpm2, -xpm1 * ypn2);
                    q1 = fma(-zmp1, xpm2, xpm1 * zmp2);
                } else
                    //Q1k - third row
                    q0 = ypn1, q1 = -zmp1, q2 = T(0), q3 = xmm1;
            }
            break;
        case 32: //ymn1, xpm2
            //Q1 - third row
            a = fma(xpm2, ymn1, -xpm1 * ymn2);
            b = fma(xpm2, zpp1, -xpm1 * zpp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn2, ymn1, fma(zmp2, zpp1, xmm2 * xpm1));
                q3 = fma(zpp1, -ymn2, zpp2 * ymn1);
            } else {
                //Q1 - first row
                a = fma(ymn2, zmp1, -ymn1 * zmp2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm1, zmp2, -xmm2 * zmp1);
                    q3 = fma(xmm1, -ymn2, xmm2 * ymn1);
                } else //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 33: //ymn1, ypn2
            //Q1 - third row
            a = fma(xmm2, xpm1, fma(ypn2, ymn1, zmp2 * zpp1));
            b = fma(xpm2, zpp1, -xpm1 * zpp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm2, ymn1, -ymn2 * xpm1);
                q3 = fma(zpp2, ymn1, -ymn2 * zpp1);
            } else {
                //Q1 - first row
                a = fma(-xmm2, zmp1, xmm1 * zmp2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(-zmp2, ymn1, ymn2 * zmp1);
                    q3 = fma(xmm2, ymn1, -ymn2 * xmm1);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 34: //ymn1, zpp2
            //Q1 - third row
            a = fma(-ymn2, zpp1, ymn1 * zpp2);
            b = fma(xpm2, zpp1, -xpm1 * zpp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm2, ymn1, -xpm1 * ymn2);
                q2 = fma(xmm2, xpm1, fma(ypn2, ymn1, zmp2 * zpp1));
            } else {
                //Q1 - first row
                a = fma(xmm2, ymn1, -xmm1 * ymn2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(ymn2, zmp1, -zmp2 * ymn1);
                    q2 = fma(-xmm2, zmp1, xmm1 * zmp2);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 35: //ymn1, xmm2
            //Q1 - first row
            a = fma(xmm2, ymn1, -xmm1 * ymn2);
            b = fma(xmm2, zmp1, -xmm1 * zmp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q2 = -b, q3 = a;
                q0 = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                q1 = fma(ymn2, zmp1, -ymn1 * zmp2);
            } else {
                //Q1 - third row
                a = fma(ymn2, -zpp1, ymn1 * zpp2);
                b = fma(xmm2, xpm1, fma(ypn2, ymn1, zmp2 * zpp1)); //drop negative since negating below
                if (fabs(a) > tol || fabs(b) > tol) {
                    q2 = b, q3 = a;
                    q0 = fma(-zpp2, xpm1, xpm2 * zpp1);
                    q1 = fma(-ymn2, xpm1, xpm2 * ymn1);
                } else
                    //Q2k - third row
                    q0 = ypn2, q1 = -zmp2, q2 = T(0), q3 = xmm2;
            }
            break;
        case 36: //ymn1, ymn2
            //Q2 - third row
            a = fma(-ymn1, zpp2, ymn2 * zpp1);
            b = fma(-xpm1, ymn2, xpm2 * ymn1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q1 = -b, q3 = a;
                q0 = fma(xpm1, zpp2, -xpm2 * zpp1);
                q2 = fma(zmp1, zpp2, fma(xmm1, xpm2, ypn1 * ymn2));
            } else {
                //Q2 - first row
                a = fma(xmm1, ymn2, -xmm2 * ymn1);
                b = fma(-ymn1, zmp2, ymn2 * zmp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q1 = -b, q3 = a;
                    q0 = fma(xpm1, xmm2, fma(zpp1, zmp2, ypn1 * ymn2));
                    q2 = fma(xmm2, zmp1, -xmm1 * zmp2);
                } else
                    //Q1k - fourth row
                    q0 = zpp1, q1 = ymn1, q2 = -xmm1, q3 = T(0);
            }
            break;
        case 37: //ymn1, zmp2
            //Q2 - first row
            a = fma(xmm1, ymn2, -xmm2 * ymn1);
            b = fma(-ymn1, zmp2, ymn2 * zmp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q1 = -b, q3 = a;
                q0 = fma(xpm1, xmm2, fma(zpp1, zmp2, ypn1 * ymn2));
                q2 = fma(xmm2, zmp1, -xmm1 * zmp2);
            } else {
                //Q2 - fourth row
                a = fma(xmm1, xpm2, fma(ymn1, ypn2, zpp1 * zmp2));
                b = fma(-xpm1, zmp2, xpm2 * zmp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q1 = b, q3 = -a;
                    q0 = fma(xpm1, ypn2, -ypn1 * xpm2);
                    q2 = fma(ypn2, zmp1, -ypn1 * zmp2);
                } else
                    //Q1k - fourth row
                    q0 = zpp1, q1 = ymn1, q2 = -xmm1, q3 = T(0);
            }
            break;
        case 40: //zmp1, xpm2
            //Q1 - fourth row
            a = fma(xpm2, zmp1, -xpm1 * zmp2);
            b = fma(-xpm2, ypn1, xpm1 * ypn2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q1 = a;
                q2 = fma(ypn2, zmp1, -ypn1 * zmp2);
                q3 = fma(ymn2, ypn1, fma(zpp2, zmp1, xmm2 * xpm1));
            } else {
                //Q1 - first row
                a = fma(ymn2, zmp1, -ymn1 * zmp2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q1 = a;
                    q2 = fma(xmm1, zmp2, -xmm2 * zmp1);
                    q3 = fma(xmm1, -ymn2, xmm2 * ymn1);
                } else //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 41: //zmp1, ypn2
            //Q1 - fourth row
            a = fma(ypn2, zmp1, -ypn1 * zmp2);
            b = fma(xpm2, -ypn1, xpm1 * ypn2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q2 = a;
                q1 = fma(xpm2, zmp1, -xpm1 * zmp2);
                q3 = fma(xpm1, xmm2, fma(zpp2, zmp1, ymn2 * ypn1));
            } else {
                //Q1 - first row
                a = fma(-xmm2, zmp1, xmm1 * zmp2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q2 = a;
                    q1 = fma(-zmp2, ymn1, ymn2 * zmp1);
                    q3 = fma(xmm2, ymn1, -ymn2 * xmm1);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 42: //zmp1, zpp2
            //Q1 - fourth row
            a = fma(xmm2, xpm1, fma(ymn2, ypn1, zpp2 * zmp1));
            b = fma(xpm2, -ypn1, xpm1 * ypn2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q0 = b, q3 = a;
                q1 = fma(xpm2, zmp1, -zmp2 * xpm1);
                q2 = fma(ypn2, zmp1, -zmp2 * ypn1);
            } else {
                //Q1 - first row
                a = fma(xmm2, ymn1, -xmm1 * ymn2);
                b = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                if (fabs(a) > tol || fabs(b) > tol) {
                    q0 = b, q3 = a;
                    q1 = fma(ymn2, zmp1, -zmp2 * ymn1);
                    q2 = fma(-xmm2, zmp1, xmm1 * zmp2);
                } else
                    //Q2k - first row
                    q0 = T(0), q1 = xpm2, q2 = ypn2, q3 = zpp2;
            }
            break;
        case 43: //zmp1, xmm2
            //Q1 - first row
            a = fma(xmm2, ymn1, -xmm1 * ymn2);
            b = fma(xmm2, zmp1, -xmm1 * zmp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q2 = -b, q3 = a;
                q0 = fma(xpm2, xmm1, fma(ypn2, ymn1, zpp2 * zmp1));
                q1 = fma(ymn2, zmp1, -ymn1 * zmp2);
            } else {
                //Q1 - fourth row
                a = fma(xmm2, xpm1, fma(ymn2, ypn1, zpp2 * zmp1));
                b = fma(-ypn2, zmp1, ypn1 * zmp2);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q2 = -b, q3 = a;
                    q0 = fma(ypn2, xpm1, -xpm2 * ypn1);
                    q1 = fma(-zmp2, xpm1, xpm2 * zmp1);
                } else
                    //Q2k - third row
                    q0 = ypn2, q1 = -zmp2, q2 = T(0), q3 = xmm2;
            }
            break;
        case 44: //zmp1, ymn2
            //Q1 - first row
            a = fma(xmm2, ymn1, -xmm1 * ymn2);
            b = fma(-ymn2, zmp1, ymn1 * zmp2);
            if (fabs(a) > tol || fabs(b) > tol) {
                q1 = -b, q3 = a;
                q0 = fma(xpm2, xmm1, fma(zpp2, zmp1, ypn2 * ymn1));
                q2 = fma(xmm1, zmp2, -xmm2 * zmp1);
            } else {
                //Q1 - fourth row
                a = fma(xmm2, xpm1, fma(ymn2, ypn1, zpp2 * zmp1));
                b = fma(-xpm2, zmp1, xpm1 * zmp2);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q1 = b, q3 = -a;
                    q0 = fma(xpm2, ypn1, -ypn2 * xpm1);
                    q2 = fma(ypn1, zmp2, -ypn2 * zmp1);
                } else
                    //Q2k - fourth row
                    q0 = zpp2, q1 = ymn2, q2 = -xmm2, q3 = T(0);
            }
            break;
        case 45: //zmp1, zmp2
            //Q2 - fourth row
            a = fma(ypn1, zmp2, -ypn2 * zmp1);
            b = fma(xpm1, -zmp2, xpm2 * zmp1);
            if (fabs(a) > tol || fabs(b) > tol) {
                q1 = -b, q2 = a;
                q0 = fma(xpm1, -ypn2, ypn1 * xpm2);
                q3 = fma(ymn1, ypn2, fma(xmm1, xpm2, zpp1 * zmp2));
            } else {
                //Q2 - first row
                a = fma(-xmm1, zmp2, xmm2 * zmp1);
                b = fma(-ymn1, zmp2, ymn2 * zmp1);
                if (fabs(a) > tol || fabs(b) > tol) {
                    q1 = -b, q2 = a;
                    q0 = fma(xpm1, xmm2, fma(ymn2, ypn1, zpp1 * zmp2));
                    q3 = fma(-ymn1, xmm2, xmm1 * ymn2);
                } else
                    //Q1k - second row
                    q0 = xpm1, q1 = T(0), q2 = zmp1, q3 = -ymn1;
            }
            break;
        default:
            q0 = T(1), q1 = T(0), q2 = T(0), q3 = T(0);
    }

    //whether to normalize quaternion
    if (normalize) {
        const T norm = T(1) / sqrt(fma(q0, q0, fma(q1, q1, fma(q2, q2, q3 * q3))));
        q_align[0] = q0 * norm, q_align[1] = q1 * norm, q_align[2] = q2 * norm, q_align[3] = q3 * norm;
    } else {
        q_align[0] = q0, q_align[1] = q1, q_align[2] = q2, q_align[3] = q3;
    }
}

//Degenerate case solution to 2 point Wahba's problem when either or both of the reference and target sets are collinear
//rotates weighted sum or weighted difference of one set of points to the collinear set
template <class T>
void estimateQuatSUPER_2PTDegenerate(std::array<T, 3> &refpt1, std::array<T, 3> &refpt2, std::array<T, 3> &tarpt1,
                                     std::array<T, 3> &tarpt2, T &w1, T &w2, std::array<T, 4> &q_opt) {
    constexpr T tol = std::is_same<T, double>::value ? 1e-9 : 1e-6f; //tolerance for zero checking
    const T &x1 = refpt1[0], &y1 = refpt1[1], &z1 = refpt1[2], &x2 = refpt2[0], &y2 = refpt2[1], &z2 = refpt2[2];
    const T &m1 = tarpt1[0], &n1 = tarpt1[1], &p1 = tarpt1[2], &m2 = tarpt2[0], &n2 = tarpt2[1], &p2 = tarpt2[2];

    T dot1 = fma(x1, x2, fma(y1, y2, z1 * z2));
    T dot2 = fma(m1, m2, fma(n1, n2, p1 * p2));
    bool tar_singular = fabs(dot2) > fabs(dot1);

    if (tar_singular) {
        const T xm = dot2 > 0 ? fma(w1, x1, w2 * x2) : fma(w1, x1, -w2 * x2);
        const T ym = dot2 > 0 ? fma(w1, y1, w2 * y2) : fma(w1, y1, -w2 * y2);
        const T zm = dot2 > 0 ? fma(w1, z1, w2 * z2) : fma(w1, z1, -w2 * z2);
        T xm_norm = sqrt(fma(xm, xm, fma(zm, zm, ym * ym)));
        if (xm_norm > tol) {
            xm_norm = T(1) / xm_norm;
            std::array<T, 3> xm_vec = {xm * xm_norm, ym * xm_norm, zm * xm_norm};
            alignQuatSUPER(xm_vec, tarpt1, q_opt);
        } else {
            T norm = sqrt(fma(-z1, z1, T(1)));
            if (norm > tol) {
                norm = T(1) / norm;
                std::array<T, 3> norm_vec = {-y1 * norm, x1 * norm, T(0)};
                alignQuatSUPER(norm_vec, tarpt1, q_opt);
            } else {
                norm = T(1) / sqrt(fma(-y1, y1, T(1)));
                std::array<T, 3> norm_vec = {-z1 * norm, T(0), x1 * norm};
                alignQuatSUPER(norm_vec, tarpt1, q_opt);
            }
        }
    } else {
        const T xm = dot1 > 0 ? fma(w1, m1, w2 * m2) : fma(w1, m1, -w2 * m2);
        const T ym = dot1 > 0 ? fma(w1, n1, w2 * n2) : fma(w1, n1, -w2 * n2);
        const T zm = dot1 > 0 ? fma(w1, p1, w2 * p2) : fma(w1, p1, -w2 * p2);
        T xm_norm = sqrt(fma(xm, xm, fma(zm, zm, ym * ym)));
        if (xm_norm > tol) {
            xm_norm = T(1) / xm_norm;
            std::array<T, 3> xm_vec = {xm * xm_norm, ym * xm_norm, zm * xm_norm};
            alignQuatSUPER(refpt1, xm_vec, q_opt);
        } else {
            T norm = fma(-p1, p1, T(1));
            if (norm > tol) {
                norm = T(1) / sqrt(fma(-p1, p1, T(1)));
                std::array<T, 3> norm_vec = {-n1 * norm, m1 * norm, T(0)};
                alignQuatSUPER(refpt1, norm_vec, q_opt);
            } else {
                norm = T(1) / sqrt(fma(-n1, n1, T(1)));
                std::array<T, 3> norm_vec = {-p1 * norm, T(0), m1 * norm};
                alignQuatSUPER(refpt1, norm_vec, q_opt);
            }
        }
    }
}

///////////////Wahba's Problem - 2 point closed form solutions

/**
 * 2 point closed-form solution to Wahba's problem
 *
 * Solution finds weighted average rotation between the two rotations that each align
 * the reference and target cross products, along with one of the observations
 *
 * @param refpt1    First reference point, unit norm
 * @param refpt2    Second reference point, unit norm
 * @param tarpt1    First target point, unit norm
 * @param tarpt2    Second target point, unit norm
 * @param w1        Weight of first point correspondence
 * @param w2        Weight of second point correspondence
 * @param q_opt     Optimal quaternion result
 */
template <class T>
void estimateQuatSUPER_2PT(std::array<T, 3> &refpt1, std::array<T, 3> &refpt2, std::array<T, 3> &tarpt1,
                           std::array<T, 3> &tarpt2, T w1, T w2, std::array<T, 4> &q_opt) {
    constexpr T tol = std::is_same<T, double>::value ? 1e-3 : 1e-3f; //tolerance for rotation estimation
    constexpr T tol2 = std::is_same<T, double>::value ? 1e-10 : 1e-6; //tolerance for degenerate condition
    constexpr T tol3 = std::is_same<T, double>::value ? 1e-20 : 1e-10; //tolerance for average rotation norm
    constexpr bool normalize = false;

    const T &x1 = refpt1[0], &y1 = refpt1[1], &z1 = refpt1[2], &x2 = refpt2[0], &y2 = refpt2[1], &z2 = refpt2[2];
    const T &m1 = tarpt1[0], &n1 = tarpt1[1], &p1 = tarpt1[2], &m2 = tarpt2[0], &n2 = tarpt2[1], &p2 = tarpt2[2];

    //take cross products
    T nrx = fma(y1, z2, -y2 * z1), nry = fma(z1, x2, -x1 * z2), nrz = fma(x1, y2, -y1 * x2);
    T ntx = fma(n1, p2, -p1 * n2), nty = fma(p1, m2, -m1 * p2), ntz = fma(m1, n2, -n1 * m2);
    const T nt_norm2 = fma(ntx, ntx, fma(nty, nty, ntz * ntz));
    const T nr_norm2 = fma(nrx, nrx, fma(nry, nry, nrz * nrz));

    //one of the point pairs is parallel or antiparallel
    if (nt_norm2 < tol2 || nr_norm2 < tol2) {
        estimateQuatSUPER_2PTDegenerate(refpt1, refpt2, tarpt1, tarpt2, w1, w2, q_opt);
        return;
    }

    //make cross products have the same magnitude
    if (nt_norm2 > nr_norm2) {
        const T norm_adj = sqrt(nt_norm2 / nr_norm2);
        nrx *= norm_adj, nry *= norm_adj, nrz *= norm_adj;
    } else {
        const T norm_adj = sqrt(nr_norm2 / nt_norm2);
        ntx *= norm_adj, nty *= norm_adj, ntz *= norm_adj;
    }

    bool n_normalized = false;

    //find first exact rotation aligning r1 to t1 and cross products n1 to n2
    T q1_2, q1_3;
    T xpm = nrx + ntx, ypn = -(nry + nty), pmz = ntz - nrz;
    T xpm1 = x1 + m1, ypn1 = -(y1 + n1), pmz1 = p1 - z1;
    T q1_1 = fma(xpm, pmz1, -pmz * xpm1);
    T q1_0 = fma(ypn, xpm1, -xpm * ypn1);
    //test first if a standard rotation estimate is sufficiently far from singularity
    if (fabs(xpm) > tol && (fabs(ypn1) > tol || fabs(pmz1) > tol) && fabs(q1_0) > tol && fabs(q1_1) > tol) {
        q1_2 = fma(pmz, ypn1, -ypn * pmz1);
        q1_3 = fma(nrz + ntz, pmz1, fma(nry - nty, ypn1, (ntx - nrx) * xpm1));
    } else {
        //estimate might be singular or points may be near parallel/antiparallel
        //default back to robust rotation estimate

        //normalization optional but may improve stability when near parallel/antiparallel
//        if (nt_norm2 > nr_norm2) {
//            const T nt_norm = T(1) / sqrt(nt_norm2);
//            ntx *= nt_norm, nty *= nt_norm, ntz *= nt_norm;
//            nrx *= nt_norm, nry *= nt_norm, nrz *= nt_norm;
//        } else {
//            const T nr_norm = T(1) / sqrt(nr_norm2);
//            ntx *= nr_norm, nty *= nr_norm, ntz *= nr_norm;
//            nrx *= nr_norm, nry *= nr_norm, nrz *= nr_norm;
//        }
//        n_normalized = true;
        xpm = nrx + ntx, ypn = -(nry + nty), pmz = ntz - nrz;
        std::array<T, 3> nr = {nrx, nry, nrz};
        std::array<T, 3> nt = {ntx, nty, ntz};
        alignQuatSUPER_2PT(nr, refpt1, nt, tarpt1, q_opt, false); //normalization optional
        q1_0 = q_opt[0], q1_1 = q_opt[1], q1_2 = q_opt[2], q1_3 = q_opt[3];
    }

    //find second exact rotation aligning r2 to t2 and cross products n1 to n2
    T q2_2, q2_3;
    T xpm2 = x2 + m2, ypn2 = -(y2 + n2), pmz2 = p2 - z2;
    T q2_1 = fma(xpm, pmz2, -pmz * xpm2);
    T q2_0 = fma(ypn, xpm2, -xpm * ypn2);
    //test first if a standard rotation estimate is sufficiently far from singularity
    if (fabs(xpm) > tol && (fabs(ypn2) > tol || fabs(pmz2) > tol) && fabs(q2_0) > tol && fabs(q2_1) > tol) {
        q2_2 = fma(pmz, ypn2, -ypn * pmz2);
        q2_3 = fma(nrz + ntz, pmz2, fma(nry - nty, ypn2, (ntx - nrx) * xpm2));
    } else {
        //estimate might be singular or points may be near parallel/antiparallel
        //default back to robust rotation estimate

        //normalization optional but may improve stability when near parallel/antiparallel
//        if (!n_normalized) {
//            if (nt_norm2 > nr_norm2) {
//                const T nt_norm = T(1) / sqrt(nt_norm2);
//                ntx *= nt_norm, nty *= nt_norm, ntz *= nt_norm;
//                nrx *= nt_norm, nry *= nt_norm, nrz *= nt_norm;
//            } else {
//                const T nr_norm = T(1) / sqrt(nr_norm2);
//                ntx *= nr_norm, nty *= nr_norm, ntz *= nr_norm;
//                nrx *= nr_norm, nry *= nr_norm, nrz *= nr_norm;
//            }
//        }
        std::array<T, 3> nr2 = {nrx, nry, nrz};
        std::array<T, 3> nt2 = {ntx, nty, ntz};
        alignQuatSUPER_2PT(nr2, refpt2, nt2, tarpt2, q_opt, false); //normalization optional
        q2_0 = q_opt[0], q2_1 = q_opt[1], q2_2 = q_opt[2], q2_3 = q_opt[3];
    }

    //norm and dot product of rotations
    T norm_q1_2 = fma(q1_0, q1_0, fma(q1_1, q1_1, fma(q1_2, q1_2, q1_3 * q1_3)));
    T norm_q2_2 = fma(q2_0, q2_0, fma(q2_1, q2_1, fma(q2_2, q2_2, q2_3 * q2_3)));
    const T q1_dot_q2 = fma(q1_0, q2_0, fma(q1_1, q2_1, fma(q1_2, q2_2, q1_3 * q2_3)));

    //solve for weights a*q_1 + b*q_2 to make average rotation
    const T AB01 = T(2) * w1 * norm_q2_2 * q1_dot_q2;
    const T AB10 = T(2) * w2 * norm_q1_2 * q1_dot_q2;
    T a, b;
    if (w1 > w2) {
        const T ad = (w1 - w2) * norm_q2_2 * norm_q1_2;
        a = ad + sqrt(fma(ad, ad, AB01 * AB10));
        b = AB10;
    } else {
        const T w_diff = w2 - w1;
        const T ad = w_diff * norm_q2_2 * norm_q1_2;
        a = AB01;
        b = ad + sqrt(fma(ad, ad, AB01 * AB10));
        if (fabs(q1_dot_q2) < tol2 && w_diff < tol2) //no unique solution
            a = T(0), b = T(1);
    }
    //normalize weights so sum has unit norm
    T ab_norm = sqrt(fma(norm_q1_2 * a, a, b * fma(norm_q2_2, b, T(2) * a * q1_dot_q2)));
    if (ab_norm > tol3) {
        ab_norm = T(1) / ab_norm;
        a *= ab_norm, b *= ab_norm;

        //average
        q_opt[0] = fma(a, q1_0, b * q2_0);
        q_opt[1] = fma(a, q1_1, b * q2_1);
        q_opt[2] = fma(a, q1_2, b * q2_2);
        q_opt[3] = fma(a, q1_3, b * q2_3);
    } else {
        //something's wrong, norm of the quaternions being averaged is too low
        //considering raising tol, lowering tol3, or normalizing quaternion
        //for now, take greater weight rotation
        std::cout << "Warning: reached edge case in weighted estimateQuatSUPER_2PT where rotation average is 0" << std::endl;
        norm_q1_2 = sqrt(norm_q1_2);
        norm_q2_2 = sqrt(norm_q2_2);
        if (w2 > w1 && norm_q1_2 > tol3) {
            a = T(1) / norm_q1_2;
            q_opt[0] = a * q1_0, q_opt[1] = a * q1_1, q_opt[2] = a * q1_2, q_opt[3] = a * q1_3;
        } else {
            if (norm_q2_2 > tol) {
                b = T(1) / norm_q2_2;
                q_opt[0] = b * q2_0, q_opt[1] = b * q2_1, q_opt[2] = b * q2_2, q_opt[3] = b * q2_3;
            } else {
                q_opt[0] = T(1), q_opt[1] = T(0), q_opt[2] = T(0), q_opt[3] = T(0);
            }
        }
    }
}

/**
 * 2 point closed-form solution to Wahba's problem for unweighted observations
 *
 * Solution finds the rotation that exactly aligns (refpt1 - refpt2) to (tarpt1 - tarpt2),
 * and (ref1 + ref2) to (tar1 + tar2)
 *
 * @param refpt1    First reference point, unit norm
 * @param refpt2    Second reference point, unit norm
 * @param tarpt1    First target point, unit norm
 * @param tarpt2    Second target point, unit norm
 * @param q_opt     Optimal quaternion result
 */
template <class T>
void estimateQuatSUPER_2PT(std::array<T, 3> &refpt1, std::array<T, 3> &refpt2, std::array<T, 3> &tarpt1,
                           std::array<T, 3> &tarpt2, std::array<T, 4> &q_opt) {
    constexpr T tol = std::is_same<T, double>::value ? 5e-4 : 1e-3f; //tolerance for rotation estimation
    constexpr T tol2 = std::is_same<T, double>::value ? 1e-10 : 1e-6f; //tolerance for degenerate condition
    const T &x1 = refpt1[0], &y1 = refpt1[1], &z1 = refpt1[2], &x2 = refpt2[0], &y2 = refpt2[1], &z2 = refpt2[2];
    const T &m1 = tarpt1[0], &n1 = tarpt1[1], &p1 = tarpt1[2], &m2 = tarpt2[0], &n2 = tarpt2[1], &p2 = tarpt2[2];

    //calculate norm via dot product
    const T dot1 = fma(x1, x2, fma(y1, y2, z1 * z2));
    const T dot2 = fma(m1, m2, fma(n1, n2, p1 * p2));
    const T abp_norm1 = T(1) + dot1, abp_norm2 = T(1) + dot2;
    const T abm_norm1 = T(1) - dot1, abm_norm2 = T(1) - dot2;

    //check if degenerate
    if (fabs(abp_norm1) < tol2 || fabs(abp_norm2) < tol2 || fabs(abm_norm1) < tol2 || fabs(abm_norm2) < tol2) {
        T w1 = 1., w2 = 1.;
        estimateQuatSUPER_2PTDegenerate(refpt1, refpt2, tarpt1, tarpt2, w1, w2, q_opt);
        return;
    }

    //calculate sums and differences
    T sx = x1 + x2, sy = y1 + y2, sz = z1 + z2;
    T sm = m1 + m2, sn = n1 + n2, sp = p1 + p2;
    T dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
    T dm = m1 - m2, dn = n1 - n2, dp = p1 - p2;

    //adjust sum and difference vectors respectively to same magnitude
    if (abp_norm1 > abp_norm2) {
        const T norm_adj = sqrt(abp_norm1 / abp_norm2);
        sm *= norm_adj, sn *= norm_adj, sp *= norm_adj;
    } else {
        const T norm_adj = sqrt(abp_norm2 / abp_norm1);
        sx *= norm_adj, sy *= norm_adj, sz *= norm_adj;
    }
    if (abm_norm1 > abm_norm2) {
        const T norm_adj = sqrt(abm_norm1 / abm_norm2);
        dm *= norm_adj, dn *= norm_adj, dp *= norm_adj;
    } else {
        const T norm_adj = sqrt(abm_norm2 / abm_norm1);
        dx *= norm_adj, dy *= norm_adj, dz *= norm_adj;
    }

    //find exact rotation aligning a1+a2 to b1+b2 and a1-a2 to b1-b2
    T xpm1 = sx + sm, ypn1 = -(sy + sn), pmz1 = sp - sz;
    T xpm2 = dx + dm, ypn2 = -(dy + dn), pmz2 = dp - dz;
    T a = fma(xpm1, pmz2, -pmz1 * xpm2);
    T b = fma(ypn1, xpm2, -xpm1 * ypn2);
    //test first if a standard rotation estimate is sufficiently far from singularity
    if (fabs(xpm1) > tol && (fabs(ypn2) > tol || fabs(pmz2) > tol) && fabs(a) > tol && fabs(b) > tol) {
        const T q2 = fma(pmz1, ypn2, -ypn1 * pmz2);
        const T q3 = fma(sz + sp, pmz2, fma(sy - sn, ypn2, (sm - sx) * xpm2));

        //normalize
        const T norm = T(1) / sqrt(fma(a, a, fma(b, b, fma(q2, q2, q3 * q3))));
        q_opt[0] = b * norm, q_opt[1] = a * norm, q_opt[2] = q2 * norm, q_opt[3] = q3 * norm;
    } else {
        //estimate might be singular or points may be near parallel/antiparallel
        //default back to robust rotation estimate

        //normalization optional but may improve stability when near parallel/antiparallel
//        const T abp_norm = T(1) / sqrt(2 * (abp_norm1 > abp_norm2 ? abp_norm1 : abp_norm2));
//        sx *= abp_norm, sy *= abp_norm, sz *= abp_norm;
//        sm *= abp_norm, sn *= abp_norm, sp *= abp_norm;
//        const T abm_norm = T(1) / sqrt(2 * (abm_norm1 > abm_norm2 ? abm_norm1 : abm_norm2));
//        dx *= abm_norm, dy *= abm_norm, dz *= abm_norm;
//        dm *= abm_norm, dn *= abm_norm, dp *= abm_norm;

        std::array<T, 3> s1 = {sx, sy, sz};
        std::array<T, 3> s2 = {sm, sn, sp};
        std::array<T, 3> d1 = {dx, dy, dz};
        std::array<T, 3> d2 = {dm, dn, dp};
        alignQuatSUPER_2PT(s1, d1, s2, d2, q_opt, true);
    }
}

///////////////Wahba's Problem - General case
//Note: Matrix A below is the objective matrix which is denoted G in paper (G_P, G_S).

// Solution options
// FAST is approximation to eigenvector solution as [-det(A_{1:,1:}), A_{1:,1:}^{-1}A_{0, 1:}], i.e. assumes A's eigenvalue is 0
// ROBUST is FAST algorithm but handles 180 degree rotations and finds best axis to project for approximation
// EIGVEC runs ROBUST to create matrix A but obtains true eigenvector via Jacobi eigenvalue algorithm (slowest)
// MOBIUS runs Mobius approximation which is in different method (flag used for testing logic)
enum solve_options {FAST = 0, ROBUST = 1, EIGVEC = 2, MOBIUS=3};

/**
 * General case solutions to Wahba's problem derived from special unitary matrices.
 * stereographic flag determines whether solution uses G_P or G_S from paper
 *
 * @tparam solve_option     Which option in solve_options above to run. Singular for 180 degree rotations
 * @tparam stereographic    Whether inputs are 2D stereographic coordinates or 3D unit norm input vectors
 * @tparam iters            Number of iterations run to refine approximation for flags FAST and ROBUST
 * @tparam perform_checks   Runs checks on inputs, normalizing weights and 3D inputs
 * @param refpts            Reference points, 2D stereographic or 3D unit vectors
 * @param tarpts            Target points, 2D stereographic or 3D unit vectors
 * @param weights           Weights corresponding to each observation. If empty and perform_checks=true, these will be filled with a constant
 * @param q_opt             Optimal quaternion returned by the algorithm. If stereographic=true, quaternion operates in stereographic plane only
 */
template <class T, int solve_option, bool stereographic=false, int iters = 2, bool perform_checks=false>
void estimateQuatSUPER(std::vector<std::array<T, stereographic ? 2 : 3>> &refpts, std::vector<std::array<T, stereographic ? 2 : 3>> &tarpts,
                       std::vector<T> &weights, std::array<T, 4> &q_opt) {

    constexpr T SUPER_TOL = 1e-3; //tolerance for checking if a reference or target point is near singularity
    const int num_pts = (int)refpts.size();

#if !JACOBI_PD
    if (solve_option == EIGVEC)
        throw std::runtime_error("Need JACOBI_PD header files and flag defined for explicit eigenvector solution.");
#endif
    if (solve_option == MOBIUS) throw std::runtime_error("For Mobius approximation call estimateMobiusSUPER() instead.");

    if (perform_checks) {
        if (iters > 2) std::cerr << "Warning: large number of iterations can result in large numerical instability";

        //perform checks
        if (num_pts < 1)
            throw std::runtime_error("Number of points must be >= 1");
        if (num_pts != tarpts.size())
            throw std::runtime_error("Number of reference and target points must be the same");
        if (!weights.empty() && weights.size() != num_pts)
            throw std::runtime_error("Number of weights provided must be 0 or equal to number of reference points");

        //normalize vectors to unit sphere
        if (!stereographic) {
            for (auto &refpt: refpts) {
                T &v0 = refpt[0], &v1 = refpt[1], &v2 = refpt[2];
                T vnorm = fma(v0, v0, fma(v1, v1, v2 * v2));
                if (fabs(vnorm) < (std::is_same<T, double>::value ? DBL_EPSILON : FLT_EPSILON))
                    throw std::runtime_error("Attempted divide by 0 during normalization");
                vnorm = T(1) / sqrt(vnorm);
                v0 *= vnorm, v1 *= vnorm, v2 *= vnorm;
            }
            for (auto &tarpt: tarpts) {
                T &v0 = tarpt[0], &v1 = tarpt[1], &v2 = tarpt[2];
                T vnorm = fma(v0, v0, fma(v1, v1, v2 * v2));
                if (fabs(vnorm) < (std::is_same<T, double>::value ? DBL_EPSILON : FLT_EPSILON))
                    throw std::runtime_error("Attempted divide by 0 during normalization");
                vnorm = T(1) / sqrt(vnorm);
                v0 *= vnorm, v1 *= vnorm, v2 *= vnorm;
            }
        }

        //create or normalize weights
        if (weights.empty()) {
            weights.reserve(num_pts);
            T weight = T(1.0) / num_pts;
            for (int i = 0; i < num_pts; i++)
                weights.push_back(weight);
        } else {
            T sum = 0;
            for (auto &weight: weights) {
                if (weight <= 0)
                    throw std::runtime_error("Weights must be positive");
                sum += weight;
            }
            sum = 1 / sum;
            for (auto &weight: weights)
                weight *= sum;
        }
    }

    //calculate symmetric matrix A^T * A
    T A00 = T(0.), A01 = T(0.), A02 = T(0.), A03 = T(0.), A11 = T(0.);
    T A12 = T(0.), A13 = T(0.), A22 = T(0.), A23 = T(0.), A33 = T(0.);
    if (!stereographic) {
        //3D sphere points, 0.5 * G_S in paper
        for (int i=0; i<num_pts; i++) {
            const T &x = refpts[i][0], &y = refpts[i][1], &z = refpts[i][2];
            const T &m = tarpts[i][0], &n = tarpts[i][1], &p = tarpts[i][2];
            const T &w = weights[i];
            const T w_half = w * T(0.5);
            const T xmm = x - m, ymn = y - n, zmp = z - p;
            const T xpm = x + m, ypn = y + n, zpp = z + p;
            const T xm2 = xmm * xmm, yn2 = ymn * ymn, zp2 = zmp * zmp;
            const T xp2 = xpm * xpm, yp2 = ypn * ypn, zpp2 = zpp * zpp;
            const T wm = w * m, wn = w * n, wp = w * p;
            const T nz = wn * z, py = wp * y, mz = wm * z, px = wp * x, my = wm * y, nx = wn * x;
            A00 += w_half * (xm2 + yn2 + zp2);
            A11 += w_half * (xm2 + yp2 + zpp2);
            A22 += w_half * (xp2 + yn2 + zpp2);
            A33 += w_half * (xp2 + yp2 + zp2);
            A01 += nz - py, A02 += px - mz, A03 += my - nx;
            A12 -= my + nx, A13 -= mz + px, A23 -= nz + py;
        }
    } else {
        //construct matrix for coordinates in stereographic plane, 0.25 * G_P in paper
        for (int i=0; i<num_pts; i++) {
            const T &x = refpts[i][0], &y = refpts[i][1];
            const T &m = tarpts[i][0], &n = tarpts[i][1];
            const T norm = weights[i] / (fma(x, x, fma(y, y, T(1))) * fma(m, m, fma(n, n, T(1))));
            const T xmm = x - m, ymn = y - n, xpm = x + m, ypn = -y - n, my = m * y, nx = n * x;
            const T mypnx = my + nx, mxmny = fma(m, x, -n * y), mypnx2 = mypnx * mypnx;
            const T opmxny = T(1) + mxmny, ommxny = T(1) - mxmny;
            A00 += norm * fma(xmm, xmm, ymn * ymn);
            A01 += norm * (my - nx);
            A02 += norm * fma(xmm, opmxny, ymn * mypnx);
            A03 += norm * fma(xmm, mypnx, ymn * ommxny);
            A11 += norm * fma(xpm, xpm, ypn * ypn);
            A12 += norm * fma(xpm, mypnx, ypn * opmxny);
            A13 += norm * fma(xpm, ommxny, ypn * mypnx);
            A22 += norm * fma(opmxny, opmxny, mypnx2);
            A23 += norm * mypnx;
            A33 += norm * fma(ommxny, ommxny, mypnx2);
        }
        A01 *= T(2);
        A23 *= T(2);
    }

    T q0, q1, q2, q3;
    if (solve_option != EIGVEC) {
        //approximate solution
        //calculate cofactor matrix and determinant of A^T * A for inverse.
        //multiply inverse to obtain projective solution.
        //solution is scaled by determinant which comes out of normalization.

        //power iterations for eigenvector refinement
        for (int i=0; i<iters; i++) {
            T B00 = fma(A01, A01, fma(A11, A11, fma(A12, A12, A13 * A13)));
            T B01 = fma(A01, A02, fma(A11, A12, fma(A12, A22, A13 * A23)));
            T B02 = fma(A01, A03, fma(A11, A13, fma(A12, A23, A13 * A33)));
            T B11 = fma(A02, A02, fma(A12, A12, fma(A22, A22, A23 * A23)));
            T B12 = fma(A02, A03, fma(A12, A13, fma(A22, A23, A23 * A33)));
            T B22 = fma(A03, A03, fma(A13, A13, fma(A23, A23, A33 * A33)));

            T b0 = fma(A01, A00, fma(A11, A01, fma(A12, A02, A13 * A03)));
            T b1 = fma(A02, A00, fma(A12, A01, fma(A22, A02, A23 * A03)));
            T b2 = fma(A03, A00, fma(A13, A01, fma(A23, A02, A33 * A03)));

            if (iters > 1)
                A00 = fma(A00, A00, fma(A01, A01, fma(A02, A02, A03 * A03)));
            A11 = B00, A12 = B01, A13 = B02, A22 = B11;
            A23 = B12, A33 = B22, A01 = b0, A02 = b1, A03 = b2;
        }

        //project from first direction
        T sum1 = fma(A22, A33, -A23 * A23);
        T sum2 = fma(A23, A13, -A12 * A33);
        T sum3 = fma(A12, A23, -A22 * A13);
        q0 = -fma(A11, sum1, fma(A12, sum2, A13 * sum3)); //-det0
        if (solve_option == FAST) {
            //choose first dimension
            //faster but has singularity if rotation is 180 degrees
            T sum4 = fma(A12, A13, -A11 * A23);
            q1 = fma(sum1, A01, fma(sum2, A02, sum3 * A03));
            q2 = fma(sum2, A01, fma(fma(A11, A33, -A13 * A13), A02, sum4 * A03));
            q3 = fma(sum3, A01, fma(sum4, A02, fma(A11, A22, -A12 * A12) * A03));
        } else {
            //find best dimension to project on by calculating largest 3x3 principal minor determinant
            //avoids rotational singularity at 180 degrees and is more stable near singularity
            int max_det_ind = 0;
            T max_det = fabs(q0);

            //dimension 1
            T sum1_2 = fma(A11, A22, -A12 * A12);
            T sum2_2 = fma(A02, A12, -A01 * A22);
            T sum3_2 = fma(A01, A12, -A02 * A11);
            q1 = fma(A00, sum1_2, fma(A01, sum2_2, A02 * sum3_2));
            if (fabs(q1) > max_det) {
                max_det = fabs(q1);
                max_det_ind = 1;
            }

            //dimension 2
            T sum1_3 = fma(A11, A33, -A13 * A13);
            T sum2_3 = fma(A13, A03, -A01 * A33);
            T sum3_3 = fma(A01, A13, -A03 * A11);
            q2 = -fma(A00, sum1_3, fma(A01, sum2_3, A03 * sum3_3));
            if (fabs(q2) > max_det) {
                max_det = fabs(q2);
                max_det_ind = 2;
            }

            //dimension 3
            T sum2_4 = fma(A23, A03, -A02 * A33);
            T sum3_4 = fma(A02, A23, -A03 * A22);
            q3 = -fma(A00, sum1, fma(A02, sum2_4, A03 * sum3_4));
            if (fabs(q3) > max_det) {
                max_det_ind = 3;
            }

            //project along dimension with maximum principal minor
            if (max_det_ind == 0) {
                T sum4 = fma(A12, A13, -A11 * A23);
                q1 = fma(sum1, A01, fma(sum2, A02, sum3 * A03));
                q2 = fma(sum2, A01, fma(sum1_3, A02, sum4 * A03));
                q3 = fma(sum3, A01, fma(sum4, A02, sum1_2 * A03));
            } else if (max_det_ind == 1) {
                T sum4 = fma(A02, A01, -A00 * A12);
                q3 = -q1;
                q0 = fma(sum1_2, A03, fma(sum2_2, A13, sum3_2 * A23));
                q1 = fma(sum2_2, A03, fma(fma(A00, A22, -A02 * A02), A13, sum4 * A23));
                q2 = fma(sum3_2, A03, fma(sum4, A13, fma(A00, A11, -A01 * A01) * A23));
            } else if (max_det_ind == 2) {
                T sum4 = fma(A01, A03, -A00 * A13);
                q0 = fma(sum1_3, A02, fma(sum2_3, A12, sum3_3 * A23));
                q1 = fma(sum2_3, A02, fma(fma(A00, A33, -A03 * A03), A12, sum4 * A23));
                q3 = fma(sum3_3, A02, fma(sum4, A12, fma(A00, A11, -A01 * A01) * A23));
            } else { //max_det_ind == 3
                T sum4 = fma(A02, A03, -A00 * A23);
                q0 = fma(sum1, A01, fma(sum2_4, A12, sum3_4 * A13));
                q1 = q3;
                q2 = fma(sum2_4, A01, fma(fma(A00, A33, -A03 * A03), A12, sum4 * A13));
                q3 = fma(sum3_4, A01, fma(sum4, A12, fma(A00, A22, -A02 * A02) * A13));
            }
        }

        //normalize quaternion
        T norm = fma(q0, q0, fma(q1, q1, fma(q2, q2, q3 * q3)));
        if (perform_checks && fabs(norm) < (std::is_same<T, double>::value ? DBL_EPSILON : FLT_EPSILON))
            throw std::runtime_error("Attempted divide by 0 during quaternion normalization");
        norm = T(1.) / sqrt(norm);

        //no permutation of values since 3D doesn't require it and stereographic is staying in domain
        q_opt[0] = q0 * norm;
        q_opt[1] = q1 * norm;
        q_opt[2] = q2 * norm;
        q_opt[3] = q3 * norm;

    } else { //EIGVEC
#if JACOBI_PD
        //reference solution to eigenvector problem
        //best in presence of high noise or if highest accuracy is required
        jacobi_pd::Jacobi<T, std::array<T, 4> &, std::array<std::array<T, 4>, 4> &,
                const std::array<std::array<T, 4>, 4> &> ecalc(4);
        std::array<std::array<T, 4>, 4> M{};
        M[0] = {A00, A01, A02, A03};
        M[1] = {A01, A11, A12, A13};
        M[2] = {A02, A12, A22, A23};
        M[3] = {A03, A13, A23, A33};
        std::array<std::array<T, 4>, 4> evecs;
        std::array<T, 4> evals;
        int n_sweeps = ecalc.Diagonalize(M, evals, evecs, jacobi_pd::Jacobi<T, std::array<T, 4> &, std::array<std::array<T, 4>, 4> &,
                const std::array<std::array<T, 4>, 4> &>::SORT_INCREASING_EVALS, true, 50);
        q_opt = evecs[0];
#endif
    }
}

//refpts and tarpts are input points in the stereographic plane (complex numbers)
//the output is the optimal special unitary matrix (flattened) that solves Wahba's problem in the stereographic plane
//special unitary matrix corresponds to rotation in stereographic plane only
template <class T, int solve_option, int iters = 2, bool perform_checks=false>
void estimateSpecialUnitarySUPER(std::vector<std::complex<T>> &refpts, std::vector<std::complex<T>> &tarpts, std::vector<T> &weights,
                                 std::array<std::complex<T>, 4> &SU_opt) {

    //create real vectors for points
    std::vector<std::array<T, 2>> ref_real(refpts.size());
    std::vector<std::array<T, 2>> tar_real(tarpts.size());
    for (int i=0; i < refpts.size(); i++) {
        ref_real[i] = {std::real(refpts[i]), std::imag(refpts[i])};
        tar_real[i] = {std::real(tarpts[i]), std::imag(tarpts[i])};
    }
    std::array<T, 4> q_opt;
    estimateQuatSUPER<T, solve_option, true, iters, perform_checks>(ref_real, tar_real, weights, q_opt);

    //create special unitary matrix
    std::complex<T> alpha(q_opt[0], q_opt[1]);
    std::complex<T> beta(q_opt[2], q_opt[3]);
    SU_opt = {alpha, beta, -std::conj(beta), std::conj(alpha)};

}

//Mobius approximation
//returns optimal Mobius transformation aligning weighted points (stereographic distance)
//M_opt is flatted Mobius transformation result.
//if return_rotation=true, Mobius transformation is mapped to special unitary matrix and stored in M_opt
template <class T, bool perform_checks=false>
void estimateMobiusSUPER(std::vector<std::complex<T>> &refpts, std::vector<std::complex<T>> &tarpts,
                         std::vector<T> &weights, std::array<std::complex<T>, 4> &M_opt, bool return_rotation=false) {

    int num_pts = refpts.size();
    constexpr T tol = std::is_same<T, double>::value ? DBL_EPSILON : FLT_EPSILON;

#if !EIGEN && !JACOBI_PD
    if (num_pts != 3)
        throw std::runtime_error("Need EIGEN library or JACOBI_PD header files and their respective flags defined for complex eigenvector solution (i.e. num_pts != 3)");
#endif

    //check inputs
    if (perform_checks) {
        if (num_pts < 1)
            throw std::runtime_error("Number of points must be >= 1");
        if (num_pts != tarpts.size())
            throw std::runtime_error("Number of reference and target points must be the same");
        if (!weights.empty() && weights.size() != num_pts)
            throw std::runtime_error("Number of weights provided must be 0 or equal to number of reference points");

        //create or normalize weights
        if (weights.empty()) {
            weights.reserve(num_pts);
            T weight = T(1.0) / num_pts;
            for (int i = 0; i < num_pts; i++)
                weights.push_back(weight);
        } else {
            T sum = 0;
            for (auto &weight: weights) {
                if (weight <= 0)
                    throw std::runtime_error("Weights must be positive");
                sum += weight;
            }
            sum = 1 / sum;
            for (auto &weight: weights)
                weight *= sum;
        }
    }

    std::complex<T> sigma, xi, gamma, delta;
    if (num_pts == 3) {
        //for three points, can find simple closed form solution
        std::complex<T> h1_0 = tarpts[1] - tarpts[2], h1_2 = tarpts[1] - tarpts[0];
        std::complex<T> h1_1 = -tarpts[0] * h1_0, h1_3 = -tarpts[2] * h1_2;
        std::complex<T> h2_0 = refpts[1] - refpts[2], h2_2 = refpts[1] - refpts[0];
        std::complex<T> h2_1 = -refpts[0] * h2_0, h2_3 = -refpts[2] * h2_2;
        sigma = h1_3 * h2_0 - h1_1 * h2_2;
        xi = h1_3 * h2_1 - h1_1 * h2_3;
        gamma = h1_0 * h2_2 - h1_2 * h2_0;
        delta = h1_0 * h2_3 - h1_2 * h2_1;
    } else {
        //estimate best solution via complex eigendecomposition

        //construct matrix M (labeled G_M in paper)
        std::complex<T> M00 = T(0), M01 = T(0), M02 = T(0), M03 = T(0), M11 = T(0);
        std::complex<T> M12 = T(0), M13 = T(0), M22 = T(0), M23 = T(0), M33 = T(0);
        for (int i = 0; i < num_pts; i++) {
            std::complex<T> &z = refpts[i], &p = tarpts[i];
            std::complex<T> z_norm2 = z * std::conj(z);
            std::complex<T> p_norm2 = p * std::conj(p);
            T w = weights[i];// / sqrt((T(1) + std::real(z_norm2)) * (T(1) + std::real(p_norm2))); //normalization helps accuracy
            M00 += w * z_norm2;
            M01 += w * std::conj(z);
            M02 -= w * z_norm2 * p;
            M03 -= w * std::conj(z) * p;
            M11 += w;
            M12 -= w * p * z;
            M13 -= w * p;
            M22 += w * z_norm2 * p_norm2;
            M23 += w * p_norm2 * std::conj(z);
            M33 += w * p_norm2;
        }

        //find eigenvector corresponding to smallest eigenvalue
#if JACOBI_PD && !EIGEN
        std::array<std::array<std::complex<T>, 4>, 4> M{};
        M[0] = {M00, M01, M02, M03};
        M[1] = {std::conj(M01), M11, M12, M13};
        M[2] = {std::conj(M02), std::conj(M12), M22, M23};
        M[3] = {std::conj(M03), std::conj(M13), std::conj(M23), M33};
        auto m_opt = jacobi_hermitian::diagonalizeHermitian<T, 4, 100>(M);
        sigma = m_opt[0], xi = m_opt[1], gamma = m_opt[2], delta = m_opt[3];
#endif
#if EIGEN
        Eigen::Matrix4cd M;
        M << M00, M01, M02, M03,
                std::conj(M01), M11, M12, M13,
                std::conj(M02), std::conj(M12), M22, M23,
                std::conj(M03), std::conj(M13), std::conj(M23), M33;

        // Compute eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> solver(M);
        auto m_opt = solver.eigenvectors().col(0);
        sigma = m_opt(0), xi = m_opt(1), gamma = m_opt(2), delta = m_opt(3);
#endif
    }

    if (return_rotation) {
        //map Mobius transformation to "nearest" special unitary matrix algebraically

        //normalize transformation so det(M) = 1
        std::complex<T> inv_root_det = sqrt(sigma * delta - gamma * xi);
        if (perform_checks && std::real(inv_root_det * std::conj(inv_root_det)) < tol)
            throw std::runtime_error("Determinant of Mobius transformation is 0.");
        inv_root_det = T(1) / inv_root_det;
        sigma *= inv_root_det, xi *= inv_root_det, gamma *= inv_root_det, delta *= inv_root_det;

        //calculate special unitary parameters
        std::complex<T> alpha = sigma + std::conj(delta);
        std::complex<T> beta = xi - std::conj(gamma);

        //normalize so det(Q) = 1
        std::complex<T> inv_norm = sqrt(alpha * std::conj(alpha) + beta * std::conj(beta));
        if (perform_checks && std::real(inv_norm * std::conj(inv_norm)) < tol)
            throw std::runtime_error("Norm of rotation is 0.");
        inv_norm = T(1) / inv_norm;
        alpha *= inv_norm, beta *= inv_norm;
        M_opt = {alpha, beta, -std::conj(beta), std::conj(alpha)};
    } else {
        M_opt = {sigma, xi, gamma, delta};
    }

}

#endif //SUPER_HPP
