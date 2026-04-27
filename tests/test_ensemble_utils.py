import torch
from neural_lam.utils import expand_ensemble_batch, fold_ensemble_batch

def test_batched_ensemble_expansion_and_folding():
    B = 2  # Batch size
    S = 50 # Ensemble size
    T, N, F = 3, 10, 4 # Time, Nodes, Features
    
    # 1. Test deterministic state expansion (B, T, N, F) -> (B*S, T, N, F)
    init_state = torch.rand(B, T, N, F)
    expanded_state = expand_ensemble_batch(init_state, n_members=S)
    
    assert expanded_state.shape == (B * S, T, N, F)
    # Ensure grouping is correct: first S elements should all equal init_state[0]
    assert torch.allclose(expanded_state[0], init_state[0])
    assert torch.allclose(expanded_state[S - 1], init_state[0])
    assert torch.allclose(expanded_state[S], init_state[1])
    
    # 2. Test folding back (B*S, T, N, F) -> (B, S, T, N, F)
    folded_state = fold_ensemble_batch(expanded_state, n_members=S)
    
    assert folded_state.shape == (B, S, T, N, F)
    assert torch.allclose(folded_state[0, 0], init_state[0])
    assert torch.allclose(folded_state[1, 49], init_state[1])

    # 3. Test with Probabilistic Lateral Boundary Conditions (B, S, T, N, F)
    prob_lbc = torch.rand(B, S, T, N, F)
    expanded_lbc = expand_ensemble_batch(prob_lbc, n_members=S, has_ensemble_dim=True)
    
    assert expanded_lbc.shape == (B * S, T, N, F)
    # The first element of B*S should be the first member of the first batch
    assert torch.allclose(expanded_lbc[0], prob_lbc[0, 0])
    # The (S)th element should be the first member of the second batch
    assert torch.allclose(expanded_lbc[S], prob_lbc[1, 0])

    # 4. Test ambiguous case where S == T (e.g., T=3 timesteps, S=3 members)
    S_ambig = 3
    init_state_ambig = torch.rand(B, S_ambig, N, F) # Shape (B, T, N, F) where T == S
    # If has_ensemble_dim=False, it should repeat B -> B*S rather than flattening B*T
    expanded_ambig = expand_ensemble_batch(init_state_ambig, n_members=S_ambig, has_ensemble_dim=False)
    assert expanded_ambig.shape == (B * S_ambig, S_ambig, N, F)
    assert torch.allclose(expanded_ambig[0], init_state_ambig[0])
    assert torch.allclose(expanded_ambig[1], init_state_ambig[0])
    assert torch.allclose(expanded_ambig[S_ambig], init_state_ambig[1])

if __name__ == "__main__":
    test_batched_ensemble_expansion_and_folding()
    print("Test passed successfully!")

