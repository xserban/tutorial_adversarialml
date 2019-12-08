import torch


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = torch.clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()
