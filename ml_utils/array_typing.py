from typing import TypeVar, Generic
import numpy as np
import torch

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class NpArray(np.ndarray, Generic[Shape, DType]):
    """Use this to type-annotate numpy arrays, e.g. image: NpArray['H,W,3', np.uint8].
    Removes the strictness of the nptyping.NDArray type, which is too strict for our purposes.
    """


class TorchTensor(torch.Tensor, Generic[Shape, DType]):
    """Use this to type-annotate torch tensors, e.g. image: TorchTensor['H,W,3', torch.uint8].
    Removes the strictness of the nptyping.NDArray type, which is too strict for our purposes.
    """
