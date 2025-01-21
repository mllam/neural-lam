# Standard library
import copy

# Third-party
import torch

# First-party
from neural_lam.utils import BufferList


def test_bufferlist_idiv():
    """Test in-place division of bufferlist"""

    tensors_to_buffer = [i * torch.ones(5) for i in range(3)]
    tensors_for_checking = copy.deepcopy(tensors_to_buffer)
    blist = BufferList(tensors_to_buffer)

    divisor = 5.0
    div_tensors = [ten / divisor for ten in tensors_for_checking]
    div_blist = copy.deepcopy(blist)
    div_blist /= divisor
    for bl_ten, check_ten in zip(div_tensors, div_blist):
        torch.testing.assert_allclose(bl_ten, check_ten)

    multiplier = 2.0
    mult_tensors = [ten * multiplier for ten in tensors_for_checking]
    mult_blist = copy.deepcopy(blist)
    mult_blist *= multiplier
    for bl_ten, check_ten in zip(mult_tensors, mult_blist):
        torch.testing.assert_allclose(bl_ten, check_ten)
