import torch
import unittest
from sign_converter import encode_sign, decode_sign
from tqdm import tqdm


class SignConverterTest(unittest.TestCase):
    def test_encode_decode_sign_1D(self):
        for _ in tqdm(range(10)):
            # Create a random tensor
            dim1 = torch.randint(1, 10000, (1,)).item()
            tensor = torch.randn(dim1).cuda()

            # Encode the sign
            encoded_tensor = encode_sign(tensor)

            # Decode the sign
            decoded_tensor = decode_sign(encoded_tensor, tensor.shape)

            # Check if the original tensor and the decoded tensor are equal
            incorrect_ones = torch.sum(tensor.sign() != decoded_tensor.sign()).item()
            if incorrect_ones / tensor.numel() > 0:
                return False
        return True

    def test_encode_decode_sign(self):
        for _ in tqdm(range(100)):
            # Create a random tensor
            dim1 = torch.randint(1, 10000, (1,)).item()
            dim2 = torch.randint(1, 10000, (1,)).item()
            tensor = torch.randn(dim1, dim2).cuda()

            # Encode the sign
            encoded_tensor = encode_sign(tensor)

            # Decode the sign
            decoded_tensor = decode_sign(encoded_tensor, tensor.shape)

            # Check if the original tensor and the decoded tensor are equal
            incorrect_ones = torch.sum(tensor.sign() != decoded_tensor.sign()).item()
            if incorrect_ones / tensor.numel() > 1e-5:
                return False
        return True

  

if __name__ == '__main__':
    unittest.main()
