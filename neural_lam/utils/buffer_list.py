"""Module wrapper exposing a list of tensors as registered buffers."""

# Standard library
from typing import Iterator, Union, overload

# Third-party
import torch
from torch import nn


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(
        self, buffer_tensors: list[torch.Tensor], persistent: bool = True
    ) -> None:
        """
        Register a collection of tensors as buffers inside a module.

        Parameters
        ----------
        buffer_tensors : Sequence[torch.Tensor]
            Buffers to register in the order they should be indexed.
        persistent : bool, optional
            If ``True``, buffers are saved in checkpoints. Default ``True``.
        """
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    @overload
    def __getitem__(self, key: int) -> torch.Tensor:
        """Integer-indexed access overload; see the implementation below."""

    @overload
    def __getitem__(self, key: slice) -> list[torch.Tensor]:
        """Slice-indexed access overload; see the implementation below."""

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Return the buffer(s) at ``key``.

        Supports integer indexing (with Python-style negative indices)
        and slice indexing (which returns a list of tensors).

        Raises
        ------
        IndexError
            If ``key`` is an out-of-range integer.
        """
        # Unpack slice indices and call recursively for each position
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        # Support negative indexing (e.g. buffer_list[-1] -> last element)
        if key < 0:
            key += len(self)
        if not (0 <= key < len(self)):
            raise IndexError(
                f"index {key} out of range for BufferList of length {len(self)}"
            )
        return getattr(self, f"b{key}")

    def __len__(self) -> int:
        """Return the number of registered buffers."""
        return self.n_buffers

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over the registered buffers in ascending index order."""
        return (self[i] for i in range(len(self)))

    def __itruediv__(self, other: float) -> "BufferList":
        """
        Divide each element in list with other.

        Parameters
        ----------
        other : float
            The value to divide by.

        Returns
        -------
        BufferList
            The modified BufferList.
        """
        return self.__imul__(1.0 / other)

    def __imul__(self, other: float) -> "BufferList":
        """
        Multiply each element in list with other.

        Parameters
        ----------
        other : float
            The value to multiply by.

        Returns
        -------
        BufferList
            The modified BufferList.
        """
        for buffer_tensor in self:
            buffer_tensor *= other

        return self
