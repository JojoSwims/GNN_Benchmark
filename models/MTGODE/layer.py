from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq


class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        # x: (batch, dim, nodes, seq_len), A: (nodes, nodes)
        x = torch.einsum("ncwl,vw->ncvl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=1):
        super().__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor))
            )

    def forward(self, x):
        outs = [conv(x) for conv in self.tconv]
        last = outs[-1].size(3)
        outs = [t[..., -last:] for t in outs]
        return torch.cat(outs, dim=1)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super().__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0), device=self.device)
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class CGPFunc(nn.Module):
    """ODE function for the Continuous Graph Propagation block."""

    def __init__(self, c_in, c_out, init_alpha):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.x0 = None
        self.adj = None
        self.nfe = 0
        self.alpha = init_alpha
        self.nconv = nconv()
        self.out = []

    def forward(self, t, x):
        adj = self.adj + torch.eye(self.adj.size(0), device=x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)

        self.out.append(x)
        self.nfe += 1
        ax = self.nconv(x, adj_norm)
        return 0.5 * self.alpha * (ax - x)


class CGPODEBlock(nn.Module):
    def __init__(self, cgpfunc, method, step_size, rtol, atol, adjoint, perturb,
                 estimated_nfe):
        super().__init__()
        self.odefunc = cgpfunc
        self.method = method
        self.step_size = step_size
        self.adjoint = adjoint
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol
        self.mlp = linear((estimated_nfe + 1) * self.odefunc.c_in, self.odefunc.c_out)

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def set_adj(self, adj):
        self.odefunc.adj = adj

    def forward(self, x, t):
        integration_time = torch.tensor([0, t]).float().type_as(x)
        solver = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
        out = solver(
            self.odefunc, x, integration_time,
            rtol=self.rtol, atol=self.atol, method=self.method,
            options=dict(step_size=self.step_size, perturb=self.perturb),
        )

        outs = self.odefunc.out
        self.odefunc.out = []
        outs.append(out[-1])
        return self.mlp(torch.cat(outs, dim=1))


class CGP(nn.Module):
    def __init__(self, cin, cout, alpha=2.0, method="rk4", time=1.0, step_size=1.0,
                 rtol=1e-4, atol=1e-3, adjoint=False, perturb=False):
        super().__init__()
        self.c_in = cin
        self.c_out = cout
        self.alpha = alpha

        if method == "euler":
            self.integration_time = time
            self.estimated_nfe = round(time / step_size)
        elif method == "rk4":
            self.integration_time = time
            self.estimated_nfe = round(time / (step_size / 4.0))
        else:
            raise ValueError(f"Unsupported CGP solver: {method!r}")

        self.CGPODE = CGPODEBlock(
            CGPFunc(self.c_in, self.c_out, self.alpha),
            method, step_size, rtol, atol, adjoint, perturb,
            self.estimated_nfe,
        )

    def forward(self, x, adj):
        self.CGPODE.set_x0(x)
        self.CGPODE.set_adj(adj)
        return self.CGPODE(x, self.integration_time)
