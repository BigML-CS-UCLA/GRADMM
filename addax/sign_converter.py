"""
Author: Zeman Li
This file is designed to efficiently encode and decode the sign of a PyTorch tensor in a 1-bit format by utilizing `torch.uint8`. 
This approach serves as a workaround for PyTorch's lack of native support for 1-bit sign tensors. 
Initially, it converts the tensor's sign into a `torch.uint8` format for compact storage. 
Subsequently, when the sign information is needed, the stored `torch.uint8` tensor can be directly used. 
Additionally, this file facilitates the conversion of the tensor's sign from the `torch.uint8` 1-bit format back to 
its original sign in int8 format, effectively preserving the tensor's sign information.
"""
import torch
import time
import contextlib


@contextlib.contextmanager
def profile_time_memory(message: str = "", prefix=None):
    start_time = time.time()
    metrics = {}
    initial_peak_memory = 0
    for device_id in range(torch.cuda.device_count()):
        initial_peak_memory += torch.cuda.max_memory_allocated(device_id)
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_id)
    try:
        yield metrics
    finally:
        elapsed_time = time.time() - start_time
        # logger.info(f"{message} elaphsed time: {elapsed_time : .2f} s")
        if prefix is not None:
            metrics[f"{prefix}_elapsed_time"] = elapsed_time
        else:
            metrics["elapsed_time"] = elapsed_time
        print(f"{message} elapsed time: {elapsed_time : .2f} s")    
        max_memory_allocated = 0
        for device_id in range(torch.cuda.device_count()):
            max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
        print(
            f"{message} peak memory: {max_memory_allocated / 1024 ** 3: .2f} GB"
        )
        if prefix is not None:
            metrics[f"{prefix}_peak_memory"] = max_memory_allocated / 1024**3
            metrics[f"{prefix}_delta_memory"] = (
                max_memory_allocated - initial_peak_memory
            ) / 1024**3
            # logger.info(
            #     f"{message} delta memory: {metrics[f'{prefix}_delta_memory']: .2f} GB"
            # )
        else:
            metrics["peak_memory"] = max_memory_allocated / 1024**3
            metrics["delta_memory"] = (
                max_memory_allocated - initial_peak_memory
            ) / 1024**3
            # logger.info(f"{message} delta memory: {metrics['delta_memory']: .2f} GB")


@torch.no_grad()
def encode_sign(tensor: torch.Tensor) -> torch.Tensor:
    """
    Encodes the sign of a PyTorch tensor into a 1-bit format using `torch.uint8`.

    This function first determines if each element of the tensor is positive (including zero) or negative,
    and then packs these signs into the bits of a `torch.uint8` tensor, significantly reducing the memory footprint.

    Args:
        tensor (torch.Tensor): The input tensor to encode the sign.

    Returns:
        torch.Tensor: The encoded sign tensor in `torch.uint8` format, along with the original tensor shape.
    """
    positive_signs = (tensor.sign() + 1) // 2
    positive_signs = positive_signs.view(-1).to(torch.uint8)

    num_elements = positive_signs.numel()
    num_of_uint8 = (num_elements + 7) // 8  

    padded_size = num_of_uint8 * 8 - num_elements
    padded_zeros = torch.zeros(padded_size, dtype=positive_signs.dtype, device=positive_signs.device)
    positive_signs = torch.cat([positive_signs, padded_zeros])

    positive_signs = positive_signs.view(-1, 8)
    idx = torch.arange(8, device=tensor.device).unsqueeze(0).unsqueeze(0)
    packed_signs = (positive_signs.byte() << idx).sum(dim=-1)
    del positive_signs

    return packed_signs.to(torch.uint8)


@torch.no_grad()
def decode_sign(packed_signs: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Decodes the signs from a 1-bit format packed in a `torch.uint8` tensor back into a sign tensor.

    This function unpacks the bits from the `torch.uint8` tensor representing signs, converts them back to a tensor
    of 0s (negative) and 1s (non-negative), and then reshapes it to the original tensor shape.

    Args:
        packed_signs (torch.Tensor): The packed signs in `torch.uint8` format.
        original_shape (tuple): The original shape of the tensor before encoding.

    Returns:
        torch.Tensor: The decoded sign tensor, reshaped to its original shape.
    """
    packed_signs = packed_signs

    unpacked_bits = (packed_signs.unsqueeze(2) >> torch.arange(8, device=packed_signs.device)) & 1

    num_original_elements = torch.prod(torch.tensor(original_shape))
    decoded_signs = unpacked_bits.view(-1)[:num_original_elements]

    decoded_signs = decoded_signs.view(original_shape).to(torch.int8)
    decoded_signs[decoded_signs == 0] = -1

    del packed_signs
    del unpacked_bits

    return decoded_signs


def main():
    torch.cuda.init()
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_id)
    with profile_time_memory("generate tensor"):
        tensor = torch.randn(20480, 5120).cuda().to(torch.float16)
    with profile_time_memory("After generate tensor"):
        print()
    print(f"Original tensor:\n{tensor}\n")
    print(f"Original tensor Shape: {tensor.shape}\n")
    print(f"Original Element Size: {tensor.element_size()}\n")
    print(f"Original number of elements: {tensor.nelement()}\n")
    with profile_time_memory("encode tensor"):
        encoded_tensor = encode_sign(tensor)
    del tensor
    with profile_time_memory("After encode tensor"): 
        print()

    print(f"Encoded tensor:\n{encoded_tensor}\n")
    print(f"Encoded tensor Shape: {encoded_tensor.shape}\n")
    print(f"Encoded Element Size: {encoded_tensor.element_size()}\n")
    print(f"Encoded number of elements: {encoded_tensor.nelement()}\n")
    print("Encoded tensor GB: ", encoded_tensor.element_size() * encoded_tensor.nelement() / 1e9)


if __name__ == "__main__":
    main()