#################################################################################
# BSD 3-Clause License
#
# Copyright (c) 2024, Akshay Chandrasekhar
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

import torch

SQRT1_2 = 0.7071067811865476

#compatible with more backends than torch.abs()
def complex_mag(a):
    return torch.sqrt(torch.real(a * torch.conj(a)))

def complex_angle(a):
    return torch.atan2(torch.imag(a), torch.real(a))

#optimal rotation aligning coordinate frame (x-axis, y-axis) to two vectors
def map_2vec(v):
    eps = 5e-7

    b1 = v[:, 0:3]
    b2 = v[:, 3:6]
    b2_n = torch.sqrt(torch.sum(torch.square(b1), dim=1, keepdim=True) /
                      torch.clip(torch.sum(torch.square(b2), dim=1, keepdim=True), min=eps)) * b2

    bp = b1 + b2_n
    bm = b1 - b2_n

    bpn = bp / torch.clip(torch.linalg.norm(bp, dim=1, keepdim=True), min=eps)
    bmn = bm / torch.clip(torch.linalg.norm(bm, dim=1, keepdim=True), min=eps)
    u1, u2, u3, v1, v2, v3 = bmn[:, 0], bmn[:, 1], bmn[:, 2], bpn[:, 0], bpn[:, 1], bpn[:, 2]
    bc = torch.stack([u2 * v3 - u3 * v2, u3 * v1 - u1 * v3, u1 * v2 - u2 * v1], dim=1)

    R = torch.stack([(bpn + bmn) * SQRT1_2, (bpn - bmn) * SQRT1_2, bc], dim=2)
    return R

class QuadMobiusRotationSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, alg_bkwd=False, alg_fwd=True):
        #map to Hermitian matrix A
        if torch.is_complex(z):
            # N x 10
            A_vec = torch.clone(z)
            z_diag = z[:, [0, 4, 7, 9]]
            A_vec[:, [0, 4, 7, 9]] = (z_diag + torch.conj(z_diag)) * 0.5
            complex_input = True
        else:
            # N x 16
            zero = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
            A_vec = torch.cat([torch.complex(z[:, 0].view(-1, 1), zero), torch.complex(z[:, 1:7:2], z[:, 2:7:2]),
                               torch.complex(z[:, 7].view(-1, 1), zero), torch.complex(z[:, 8:12:2], z[:, 9:12:2]),
                               torch.complex(z[:, 12].view(-1, 1), zero), torch.complex(z[:, 13], z[:, 14]).view(-1, 1),
                               torch.complex(z[:, 15].view(-1, 1), zero)], dim=1)
            complex_input = False
        idx1, idx2 = torch.triu_indices(4, 4)
        A = torch.empty((z.shape[0], 4, 4), dtype=A_vec.dtype, device=A_vec.device)
        A[:, idx1, idx2] = A_vec
        A[:, idx2, idx1] = torch.conj(A_vec)

        #compute eigenvector corresponding to smallest eigenvalue
        eigvals, eigvecs = torch.linalg.eigh(A, UPLO='U')
        M_opt = eigvecs[:, :, 0] #eigenvalues returned in ascending order by eigh
        M = M_opt.view(-1, 2, 2)

        if alg_fwd:
            # Previous method. Algebraic backpropagation formulas derived from this method
            # # normalize Mobius transformation to have det(M) = 1
            # Q = M / torch.sqrt(torch.linalg.det(M)).view(-1, 1, 1) #assume nonzero
            # # find nearest special unitary matrix via quaternion
            # sigma, xi, gamma, delta = Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1]
            # q_u = torch.stack([torch.real(sigma) + torch.real(delta),
            #                    -torch.imag(xi) - torch.imag(gamma),
            #                    torch.real(xi) - torch.real(gamma),
            #                    torch.imag(sigma) - torch.imag(delta)], dim=1)
            # q = q_u / torch.linalg.norm(q_u, dim=1, keepdim=True) #assume nonzero

            #normalize Mobius transformation so nearest unitary matrix is special unitary
            conj_det = torch.conj(torch.linalg.det(M))
            det_norm = complex_mag(conj_det)
            Q = M * torch.sqrt(conj_det / (det_norm * (2 * det_norm + 1.))).view(-1, 1, 1) #assumes |M_opt| = 1 from eigh
            #map to quaternion. Note ||q|| = 1 since normalization factor built-in above
            alpha, beta = Q[:, 0, 0] + torch.conj(Q[:, 1, 1]), Q[:, 0, 1] - torch.conj(Q[:, 1, 0])
            q = torch.stack([torch.real(alpha), -torch.imag(beta), torch.real(beta), torch.imag(alpha)], dim=1)

            ctx.save_for_backward(A, M_opt, eigvals[:, 0])
        else:
            #find nearest unitary matrix via SVD
            U, S, Vh = torch.linalg.svd(M)
            Q_hat = U.bmm(Vh)

            #transform to nearest special unitary matrix and map to quaternion
            Q = Q_hat * torch.conj(torch.sqrt(torch.linalg.det(Q_hat))).view(-1, 1, 1)
            alpha, beta = Q[:, 0, 0], Q[:, 0, 1]
            q = torch.stack([torch.real(alpha), -torch.imag(beta), torch.real(beta), torch.imag(alpha)], dim=1)
            ctx.save_for_backward(A, M_opt, eigvals[:, 0], U, S, Vh)

        ctx.alg_fwd, ctx.alg_bkwd, ctx.complex_input = alg_fwd, alg_bkwd, complex_input

        return q

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.alg_fwd:
            A, M_opt, eigval = ctx.saved_tensors
        else:
            A, M_opt, eigval, u, s, vh = ctx.saved_tensors
        n = A.shape[0]
        eps = 5e-7 #threshold for stability

        if ctx.alg_bkwd:
            #calculate determinant of Mobius transformation
            M = M_opt.view(-1, 2, 2)
            root_det = torch.sqrt(torch.linalg.det(M)).view(-1, 1, 1)
            zero_det_mask = complex_mag(root_det) < eps
            root_det[zero_det_mask] = eps * torch.exp(1j * complex_angle(root_det[zero_det_mask]))
            Q = M / root_det

            #map to unnormalized quaternion
            sigma, xi, gamma, delta = Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1]
            alpha, beta = sigma + torch.conj(delta), xi - torch.conj(gamma)
            q_u = torch.stack([torch.real(alpha), -torch.imag(beta), torch.real(beta), torch.imag(alpha)], dim=1)

            #gradient from normalized quaternion to unnormalized quaternion
            q_norm = torch.clip(torch.sqrt(torch.sum(q_u * q_u, dim=1, keepdim=True)), min=eps)
            q, gn = q_u / q_norm, grad_output / q_norm
            dqn = q * torch.sum(-q * gn, dim=-1, keepdim=True) + gn

            #gradient from nearest (unnormalized) quaternion to Mobius transformation
            dq0, dq1, dq2, dq3 = dqn[:, 0], dqn[:, 1], dqn[:, 2], dqn[:, 3]
            dqdM = torch.stack([torch.complex(dq0, dq3), torch.complex(dq2, -dq1), torch.complex(-dq2, -dq1), torch.complex(dq0, -dq3)], dim=1).view(-1, 2, 2)

            #gradient from normalized Mobius transformation (det(M) = 1) to unnormalized
            dqdM /= torch.conj(root_det)
            ddet = torch.stack([Q[:, 1, 1], -Q[:, 1, 0], -Q[:, 0, 1], Q[:, 0, 0]], dim=1).view(-1, 2, 2)
            dM = torch.einsum('ijk,ilm->ijklm', ddet, Q * -0.5)
            map_grad = torch.sum(torch.conj(dM) * dqdM.view(-1, 1, 1, 2, 2), dim=[3, 4]) + dqdM

        else:
            if ctx.alg_fwd:
                u, s, vh = torch.linalg.svd(M_opt.view(-1, 2, 2))
            Q_hat = u.bmm(vh)

            #gradient from quaternion to special unitary matrix
            dq = 0.5 * grad_output
            da, db = torch.complex(dq[:, 0], dq[:, 3]), torch.complex(dq[:, 2], -dq[:, 1])
            dqdQ = torch.stack([da, db, -torch.conj(db), torch.conj(da)], dim=1).view(-1, 2, 2)

            #gradient from special unitary matrix to unitary matrix
            det = torch.sqrt(torch.linalg.det(Q_hat)).view(-1, 1, 1)
            ddet = torch.conj(torch.stack([Q_hat[:, 1, 1], -Q_hat[:, 1, 0], -Q_hat[:, 0, 1], Q_hat[:, 0, 0]], dim=1))
            dQ = torch.einsum('ij,ikl->ijkl', ddet, Q_hat * 0.5)
            dQdU = (torch.sum(dQ * torch.conj(dqdQ).view(-1, 1, 2, 2), dim=[2, 3]).view(-1, 2, 2) + dqdQ) * det

            #gradient from unitary matrix to Mobius transformation
            #i.e. backward pass through polar factor of polar decomposition of 2x2 complex matrix
            #polar factor of polar decomposition is given by UV^H for SVD(M) = USV^H
            #least-norm least squares solution to continuous Lyapunov equation
            trace_sigma = torch.sum(s, dim=1)
            sigma_outer_sum = torch.stack([2 * s[:, 0], trace_sigma, trace_sigma, 2 * s[:, 1]], dim=1).view(-1, 1, 1, 2, 2)
            dA = torch.einsum('ijk,ilm->ijmlk', u, vh)
            dM = torch.where(sigma_outer_sum >= eps, dA / sigma_outer_sum, torch.zeros_like(sigma_outer_sum))

            U, Vh = u.view(-1, 1, 1, 2, 2), vh.view(-1, 1, 1, 2, 2)
            gQ = dQdU.view(-1, 1, 1, 2, 2)
            Q_grad_real = U @ dM.mH @ Vh
            Q_grad_imag = U @ dM @ Vh
            map_grad = torch.sum(torch.conj(Q_grad_real) * gQ - Q_grad_imag * torch.conj(gQ), dim=[3, 4])

        ################

        #gradient from Mobius transformation to 4x4 Hermitian matrix
        #i.e. backward pass through Hermitian eigendecomposition for eigenvector corersponding to smallest eigenvalue
        #equations given by "On Differentiating Eigenvalues and Eigenvectors" by Jan Magnus https://www.janmagnus.nl/papers/JRM011.pdf
        b = torch.zeros((n, 5, 16), device=A.device, dtype=A.dtype)
        MP = torch.zeros((n, 5, 5), device=A.device, dtype=A.dtype)

        # Solve (\lambda I - A)X = (I - M*M^H) * dA * M instead of using Moore-Penrose pseudoinverse directly
        # Since (\lambda I - A) is singular, can make it non-singular if augment with M and M^H which are orthogonal to rows and columns and use linalg.solve
        I = torch.eye(4).to(A.device).unsqueeze(0)
        proj = I - torch.einsum('ij,ik->ijk', M_opt, torch.conj(M_opt)) #/ torch.sum(M_opt * torch.conj(M_opt), dim=1).view(-1, 1, 1) #unnecessary since eigh returns ||M_opt||=1 by convention
        b[:, :4, :] = torch.einsum('ijk,il->ijkl', proj, M_opt).view(-1, 4, 16)
        MP[:, :4, :4] = I * eigval.view(-1, 1, 1) - A
        MP[:, 4, :4] = torch.conj(M_opt)
        MP[:, :4, 4] = M_opt

        singular_mask = torch.abs(torch.real(torch.linalg.det(MP))) < eps #MP is Hermitian, so det(MP) is real
        if singular_mask.any():
            #eigenvalues are (near) non-distinct, so gradient is ill-defined. More stably handle this edge-case via lstsq SVD solution (only defined on CPU)
            X = torch.zeros((n, 5, 16), device=A.device, dtype=A.dtype)
            X[singular_mask, :4] = torch.linalg.lstsq(MP[singular_mask, :4, :4].cpu(), b[singular_mask, :4].cpu(), driver='gelsd').solution.to(A.device)
            X[~singular_mask] = torch.linalg.solve(MP[~singular_mask], b[~singular_mask])
        else:
            X = torch.linalg.solve(MP, b)
        A_grad = torch.sum(torch.conj(X[:, :4, :]) * map_grad.view(-1, 4, 1), dim=1).view(-1, 4, 4)
        A_grad += A_grad.mH

        ################

        #gradient from Hermitian matrix to network output
        idx1, idx2 = torch.triu_indices(4, 4)
        if ctx.complex_input:
            A_vec_grad = A_grad[:, idx1, idx2]
            A_vec_grad[:, [0, 4, 7, 9]] *= 0.5
        else:
            idx_list = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18]
            A_vec_grad = torch.view_as_real(A_grad[:, idx1, idx2]).view(-1, 20)[:, idx_list]
            A_vec_grad[:, [0, 7, 12, 15]] *= 0.5

        return A_vec_grad, None, None, None
