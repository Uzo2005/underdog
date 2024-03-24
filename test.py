import torch
import numpy as np
from safetensors import deserialize, safe_open
from safetensors.torch import save_file
import struct

# torch.set_printoptions(threshold = 10_000, precision = 10, sci_mode = True)
def float_to_bin(f):
  # Convert the floating-point number to its binary representation
  bits = struct.pack('>f', f)
  # Convert the binary representation to a string of '0's and '1's
  binary = ''.join(f'{byte:08b}' for byte in bits)
  return binary

# firstMat = np.array([[1, 2, 1], [0, 1, 0], [2, 3, 4]])
# print(firstMat)
# print(firstMat @ firstMat @ firstMat @ firstMat @ firstMat)

SEE = 6
with safe_open("./stablelm_1_6b_model/stable_lm_1_6b_8bdf317e2b35ab5c8009cbb6c7ce495e4e608a6b9b843d44054edf25b8c5860d.safetensors", framework="pt", device="cpu") as f:
  tensor1 = f.get_tensor('lm_head.weight')
  tensor2 = f.get_tensor('model.norm.bias')
  # print(tensor2.T)
  print(tensor1.matmul(tensor2)[SEE - 1].item())
  # print(float_to_bin(tensor1.matmul(tensor2)[2].item()))
  # print(float_to_bin(tensor[0].item()))


# def read_chars_from_position(file_path, position, n):
#     try:
#         with open(file_path, 'rb') as file:
#             # Move to the specified position in the file
#             file.seek(position)
#             # Read the next n characters from that position
#             chars = file.read(n)
#             char_list = [byte for byte in chars]
#             return char_list
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#     except IOError:
#         print(f"Error: Unable to read file '{file_path}'.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# # Example usage:
# file_path = "./stablelm_1_6b_model/stable_lm_1_6b_8bdf317e2b35ab5c8009cbb6c7ce495e4e608a6b9b843d44054edf25b8c5860d.safetensors"  # Replace with the path to your file
# position = 3289022464  # Example position
# n = 2  # Example number of characters to read
# chars = read_chars_from_position(file_path, position, n)
# if chars:
#     print(f"Next {n} characters from position {position}: {chars}")

# # print(torch.tensor(chars, dtype = torch.uint8))
# # print(torch.tensor(chars, dtype = torch.uint8).to(torch.bfloat16))



# # # Example usage:
# # float_number = 0.0024
# # binary_representation = float_to_bin(float_number)
# # print("Binary representation of", float_number, ":", binary_representation)

