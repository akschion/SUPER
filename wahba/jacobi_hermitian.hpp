//#ifndef JACOBI_HERMITIAN_HPP
//#define JACOBI_HERMITIAN_HPP
#include <cmath>
#include <algorithm>
#include <complex>

//// Diagonalization of Hermitian matrices, returning eigenvector corresponding to smallest eigenvalue
//// Work in progress
namespace jacobi_hermitian {

    //square of complex magnitude
    template<class T>
    T abs2(std::complex<T> &c) {
        return std::real(c * std::conj(c));
    }

    // Helper function to print a matrix
    template<typename T, size_t N>
    void printMatrix(const std::array<std::array<std::complex<T>, N>, N> &matrix) {
        for (const auto &row: matrix) {
            for (const auto &elem: row) {
                std::cout << elem << "\t";
            }
            std::cout << "\n";
        }
    }

    //finds the largest (by complex squared magnitude) element of matrix
    //only searches lower triangular
    template<typename T, size_t N>
    void argMaxEntry(std::array<std::array<std::complex<T>, N>, N> &M, int &max_i, int &max_j) {
        max_i = 1, max_j = 0;
        T max_val = 0;
        for (int i = 1; i < N; ++i) {
            for (int j = 0; j < i; ++j) {
                T val = abs2(M[i][j]);
                if (val > max_val) {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }
    }

    //diagonalize a Hermitian matrix using complex generalization of Jacobi eigenvalue algorithm
    template<typename T, size_t N, int max_sweeps = 50>
    std::array<std::complex<T>, N> diagonalizeHermitian(std::array<std::array<std::complex<T>, N>, N> &M) {
        constexpr T tol = std::is_same<T, double>::value ? DBL_EPSILON : FLT_EPSILON;
        constexpr T tol2 = tol * tol;
        int num_sweeps = N * (N - 1) / 2;

        //initialize eigenvectors to identity matrix
        std::array<std::array<std::complex<T>, N>, N> V;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                V[i][j] = i == j ? 1.0 : 0.0;
            }
        }

        std::complex<T> UU, UL; //unitary transformation parameters (first column)
        std::complex<T> i_tmp; //tmp variable
        int i, j; //indices of max entry (lower triangular)

        int ni;
        for (ni = 0; ni < max_sweeps * num_sweeps; ni++) {
            argMaxEntry(M, i, j);
            if (M[i][j].real() == 0. && M[i][j].imag() == 0.) break;

            //eigendecomposition of 2x2 complex Hermitian matrix to find unitary parameters UU and UL
            //complex analogue of finding Givens rotation
            auto bnorm2 = abs2(M[i][j]);
            if (bnorm2 < tol2) {
                if (std::real(M[i][i]) >= std::real(M[j][j])) {
                    UU = 0.0;
                    UL = 1.0;
                } else {
                    UU = 1.0;
                    UL = 0.0;
                }
            } else {
                auto ca = std::real(M[j][j] - M[i][i]);
                auto v1 = ca + sqrt(ca * ca + 4 * bnorm2);
                auto v2 = 2.0 * M[i][j];
                auto v_norm = 1.0 / sqrt(abs2(v2) + v1 * v1);
                UU = v1 * v_norm;
                UL = v2 * v_norm;
            }

            //set the relevant diagonal elements
            auto cross_term = 2.0 * std::real(std::conj(UL) * UU * M[i][j]);
            i_tmp = std::complex<T>(M[i][i].real() * abs2(UL) + cross_term + M[j][j].real() * abs2(UU), 0.);
            M[j][j] = std::complex<T>(M[j][j].real() * abs2(UL) - cross_term + M[i][i].real() * abs2(UU), 0.);
            M[i][i] = i_tmp;
            M[i][j] = 0.0;

            //Multiply columns on right by U
            for (int row = i+1; row < N; row++) {
                i_tmp = M[row][i] * UL + M[row][j] * UU;
                M[row][j] = -M[row][i] * std::conj(UU) + M[row][j] * std::conj(UL);
                M[row][i] = i_tmp;
            }
            for (int row = j+1; row < i; row++) {
                M[j][row] = std::conj(M[row][j]); //save for later
                M[row][j] = -std::conj(M[i][row]) * std::conj(UU) + M[row][j] * std::conj(UL);
            }

            //Multiply rows on left by U^H
            for (int col = 0; col < j; col++) {
                i_tmp = M[i][col] * std::conj(UL) + M[j][col] * std::conj(UU);
                M[j][col] = -M[i][col] * UU + M[j][col] * UL;
                M[i][col] = i_tmp;
            }
            for (int col = j+1; col < i; col++) {
                M[i][col] = M[i][col] * std::conj(UL) + M[j][col] * std::conj(UU);
            }

            //transform eigenvectors on right by U
            for (int row = 0; row < N; row++) {
                i_tmp = V[row][i] * UL + V[row][j] * UU;
                V[row][j] = -V[row][i] * std::conj(UU) + V[row][j] * std::conj(UL);
                V[row][i] = i_tmp;
            }
        }

        std::array<T, N> eigenvals{};
        for (int ii=0; ii<N; ii++)  eigenvals[ii] = std::real(M[ii][ii]);

        T min_eigv = std::numeric_limits<T>::infinity();
        int argmin_eigv = 0;
        for (int ii=0; ii<N; ii++) {
            if (eigenvals[ii] < min_eigv) {
                min_eigv = eigenvals[ii];
                argmin_eigv = ii;
            }
        }

        return {V[0][argmin_eigv], V[1][argmin_eigv], V[2][argmin_eigv], V[3][argmin_eigv]};
    }
}
