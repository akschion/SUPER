import copy
import numpy as np

from learning.SUPER_maps import *

torch.manual_seed(1738)
np.random.seed(1738)

#options
device = 'cpu'
su = True #True uses QuadMobiusSVD
complex = False #whether inputs are complex or real
batch_size = 1000

if device == 'cpu':
    torch.use_deterministic_algorithms(True)

#create fake input tensor
if complex:
    a = np.random.random((batch_size, 20)).astype(np.float32) * 2 - 1
    a = a[:, ::2] + a[:, 1::2]*1j
else:
    a = np.random.random((batch_size, 16)).astype(np.float32) * 2 - 1
Av = torch.from_numpy(copy.deepcopy(a)).to(device)
D = torch.from_numpy(copy.deepcopy(a)).to(device)
Av.requires_grad_()
D.requires_grad_()
D.retain_grad()
Av.retain_grad()

##explicit construction with torch functions
if complex:
    A_vec = torch.clone(Av)
    z_diag = Av[:, [0, 4, 7, 9]]
    A_vec[:, [0, 4, 7, 9]] = torch.real(z_diag).to(Av.dtype)
else:
    zero = torch.zeros((Av.shape[0], 1), dtype=Av.dtype, device=Av.device)
    A_vec = torch.cat([torch.complex(Av[:, 0].view(-1, 1), zero), torch.complex(Av[:, 1:7:2], Av[:, 2:7:2]),
                       torch.complex(Av[:, 7].view(-1, 1), zero), torch.complex(Av[:, 8:12:2], Av[:, 9:12:2]),
                       torch.complex(Av[:, 12].view(-1, 1), zero), torch.complex(Av[:, 13], Av[:, 14]).view(-1, 1),
                       torch.complex(Av[:, 15].view(-1, 1), zero)], dim=1)
idx1, idx2 = torch.triu_indices(4, 4)
A = torch.empty((Av.shape[0], 4, 4), dtype=A_vec.dtype, device=A_vec.device)
A[:, idx1, idx2] = A_vec
A[:, idx2, idx1] = torch.conj(A_vec)
A.retain_grad()

#eigendecomposition
_, eigv = torch.linalg.eigh(A, UPLO='U')
M_opt = eigv[:, :, 0]

if su:
    #Mobius -> SU -> quat
    u, _, vh = torch.linalg.svd(M_opt.view(-1, 2, 2))
    U = u.bmm(vh)
    Q = U * torch.conj(torch.sqrt(torch.linalg.det(U))).view(-1, 1, 1)
    alpha, beta = Q[:, 0, 0], Q[:, 0, 1]
    q_opt1 = torch.stack([torch.real(alpha), -torch.imag(beta), torch.real(beta), torch.imag(alpha)], dim=1)
else:
    M = M_opt.view(-1, 2, 2)
    root_det = torch.sqrt(torch.linalg.det(M)).view(-1, 1, 1)
    Q = M / root_det #assume nonzero
    sigma, xi, gamma, delta = Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 0], Q[:, 1, 1]
    q1 = torch.vstack([torch.real(sigma) + torch.real(delta),
                      -torch.imag(xi) - torch.imag(gamma),
                      torch.real(xi) - torch.real(gamma),
                      torch.imag(sigma) - torch.imag(delta)]).T
    q_opt1 = q1 / torch.clip(torch.linalg.norm(q1, dim=1, keepdim=True), 1e-8)

q_opt2 = QuadMobiusRotationSolver.apply(D, not su, True)

rand_quat = torch.rand((Av.shape[0], 4), dtype=torch.float32, requires_grad=False, device=Av.device)
rand_quat /= torch.linalg.norm(rand_quat, dim=1, keepdim=True)
loss1 = torch.mean(torch.sum(q_opt1 * torch.sign(q_opt1[:, 0]).view(-1, 1), dim=1))
loss2 = torch.mean(torch.sum(q_opt2 * torch.sign(q_opt2[:, 0]).view(-1, 1), dim=1))
loss1.backward(retain_graph=True)
loss2.backward(retain_graph=True)

print('Batch Size: ' + str(batch_size) + ', Complex Input: ' + str(complex) + ', uses SVD: ' + str(su))
print('Gradients the same:', torch.allclose(Av.grad, D.grad, atol=5e-7))
print('Max Gradient Difference:', torch.max(torch.abs(Av.grad - D.grad)).item())
print('Quaternions the same:', torch.allclose(q_opt1 * torch.sign(q_opt1[:, 0]).view(-1, 1), q_opt2 * torch.sign(q_opt2[:, 0]).view(-1, 1), atol=5e-7))
print('Max Quaternion Difference:', torch.max(torch.abs(q_opt1 * torch.sign(q_opt1[:, 0]).view(-1, 1) - q_opt2 * torch.sign(q_opt2[:, 0]).view(-1, 1))).item())
