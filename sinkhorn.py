import torch

@torch.no_grad()
def floating_margin_sinkhorn(out, epsilon, margin_prior=None, max_iter=2):
    out_shifted = out - torch.max(out, dim=1, keepdim=True)[0]
    Q = torch.exp(out_shifted / epsilon).t() 
    
    B = Q.shape[1]
    K = Q.shape[0]

    Q /= (torch.sum(Q) + 1e-12)

    if margin_prior is not None:
        uniform = torch.ones_like(margin_prior) / K
        # 40% 均匀保障探索，60% 尊重经验允许长尾和剪枝
        margin_prior = 0.4 * uniform + 0.6 * margin_prior
    else:
        margin_prior = torch.ones(K, 1, device=out.device) / K

    for _ in range(max_iter):
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-12)
        Q /= B
        Q /= (torch.sum(Q, dim=1, keepdim=True) + 1e-12)
        Q *= margin_prior.view(-1, 1)

    return (Q.t() * B)