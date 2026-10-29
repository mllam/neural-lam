# Third-party
import pytest
import torch
import torch.nn.functional as F

# First-party
from neural_lam.utils import BufferList, inverse_softplus


@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_inverse_softplus_roundtrip(beta):
    """`inverse_softplus` recovers the input of `softplus`."""
    # Range chosen so softplus(x, beta) stays above the lower clamp
    # log(1+1e-6)/beta for all tested betas (clamp activates around
    # x = -13.8/beta).
    x_orig = torch.linspace(-5, 5, steps=100)
    y = F.softplus(x_orig, beta=beta)
    x_reconstructed = inverse_softplus(y, beta=beta)
    torch.testing.assert_close(x_orig, x_reconstructed)


def test_inverse_softplus_near_zero_is_finite():
    """Near-zero inputs are clamped so the log path cannot produce NaN/-Inf."""
    y_near_zero = torch.tensor([1e-7, 1e-6])
    x_near_zero = inverse_softplus(y_near_zero)
    assert torch.isfinite(x_near_zero).all()


@pytest.mark.parametrize("threshold", [20.0, 5.0])
def test_inverse_softplus_above_threshold_is_identity(threshold):
    """Values above `threshold` bypass the log path and return unchanged."""
    y_high = torch.tensor([threshold + 5.0, threshold + 30.0])
    x_high = inverse_softplus(y_high, threshold=threshold)
    torch.testing.assert_close(y_high, x_high)


def test_buffer_list_len_and_iteration():
    """BufferList reports correct length and iterates over buffers."""
    tensors = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    buf = BufferList(tensors)
    assert len(buf) == 3
    elements = list(buf)
    assert len(elements) == 3
    for elem, orig in zip(elements, tensors):
        torch.testing.assert_close(elem, orig)


def test_buffer_list_indexing():
    """BufferList supports positive and negative integer indexing."""
    t0 = torch.tensor([10.0])
    t1 = torch.tensor([20.0])
    buf = BufferList([t0, t1])
    torch.testing.assert_close(buf[0], t0)
    torch.testing.assert_close(buf[1], t1)
    torch.testing.assert_close(buf[-1], t1)
    torch.testing.assert_close(buf[-2], t0)


@pytest.mark.parametrize("out_of_bounds_index", [2, 10, -3, -5, -100])
def test_buffer_list_out_of_bounds_index_error_message(out_of_bounds_index):
    """Out-of-bounds integer access reports the exact requested index."""
    buf = BufferList([torch.tensor([1.0]), torch.tensor([2.0])])
    expected_msg = (
        f"^index {out_of_bounds_index} out of range for BufferList of length 2$"
    )
    with pytest.raises(IndexError, match=expected_msg):
        _ = buf[out_of_bounds_index]


@pytest.mark.parametrize("invalid_key", ["invalid", 1.5, [0], None])
def test_buffer_list_invalid_key_type_raises_type_error(invalid_key):
    """Non-integer, non-slice key raises a descriptive TypeError."""
    buf = BufferList([torch.tensor([1.0]), torch.tensor([2.0])])
    expected_msg = (
        f"BufferList indices must be integers or slices, "
        f"not {type(invalid_key).__name__}"
    )
    with pytest.raises(TypeError, match=expected_msg):
        _ = buf[invalid_key]


def test_buffer_list_slicing():
    """Slice access returns a list of tensors matching the requested slice."""
    tensors = [torch.tensor([float(i)]) for i in range(5)]
    buf = BufferList(tensors)
    sliced = buf[1:4:2]
    assert len(sliced) == 2
    torch.testing.assert_close(sliced[0], torch.tensor([1.0]))
    torch.testing.assert_close(sliced[1], torch.tensor([3.0]))


def test_buffer_list_imul():
    """In-place multiplication scales each buffer and returns BufferList."""
    t0 = torch.tensor([1.0, 2.0])
    t1 = torch.tensor([3.0, 4.0])
    buf = BufferList([t0, t1])
    ret = buf.__imul__(2.5)
    assert ret is buf
    torch.testing.assert_close(buf[0], torch.tensor([2.5, 5.0]))
    torch.testing.assert_close(buf[1], torch.tensor([7.5, 10.0]))


def test_buffer_list_itruediv():
    """In-place true division performs exact tensor division across buffers."""
    t0 = torch.tensor([1.0, 2.0])
    t1 = torch.tensor([3.0, 4.0])
    buf = BufferList([t0, t1])
    ret = buf.__itruediv__(3.0)
    assert ret is buf
    # Exact in-place true division avoids reciprocal multiplication error
    torch.testing.assert_close(buf[0], torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
    torch.testing.assert_close(buf[1], torch.tensor([1.0, 4.0 / 3.0]))


def test_buffer_list_itruediv_with_tensor():
    """In-place division supports scalar torch.Tensor divisors."""
    t0 = torch.tensor([6.0, 8.0])
    buf = BufferList([t0])
    buf /= torch.tensor(2.0)
    torch.testing.assert_close(buf[0], torch.tensor([3.0, 4.0]))


def test_buffer_list_persistence_flag():
    """BufferList respects persistent flag in PyTorch state dict."""
    t0 = torch.tensor([1.0])
    persistent_buf = BufferList([t0], persistent=True)
    non_persistent_buf = BufferList([t0], persistent=False)
    assert "b0" in persistent_buf.state_dict()
    assert "b0" not in non_persistent_buf.state_dict()
