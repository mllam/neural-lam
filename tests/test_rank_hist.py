import torch
from neural_lam.metrics import rank_histogram

def test_rank_histogram_shape():
    M, N, D = 10, 100, 4
    ens = torch.randn(M, N, D)
    target = torch.randn(N, D)
    
    hist = rank_histogram(ens, target)
    assert hist.shape == (M + 1, D)
    assert hist.sum().item() == N * D