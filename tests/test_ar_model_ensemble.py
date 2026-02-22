import torch
from neural_lam.models.ar_model import ARModel

def test_ar_model_ensemble():
	model = ARModel(None, None, None, output_mode="ensemble", ensemble_size=3)
	x = torch.randn(2, 10, 32)
	out = model(x)
	assert out.shape[0] == 3