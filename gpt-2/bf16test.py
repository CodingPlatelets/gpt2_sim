import torch

# generate 2 random bf16 tensors and multiply them


def generate_and_multiply_bf16_tensors():
    # Generate two random bf16 tensors
    torch.manual_seed(42)  # For reproducibility
    tensor1 = torch.randn(4, 99999, dtype=torch.bfloat16)
    tensor2 = torch.randn(99999, 4, dtype=torch.bfloat16)

    tensor16 = tensor1.clone().to('cuda')
    tensor26 = tensor2.clone().to('cuda')
    # Multiply the tensors
    result = torch.matmul(tensor1, tensor2)
    result16 = torch.matmul(tensor16, tensor26)

    return result, result16.cpu()


# Test the function
if __name__ == "__main__":
    result, result6 = generate_and_multiply_bf16_tensors()
    print("Result shape:", result.shape)
    print("Result dtype:", result.dtype)
    print("Result1:", result)
    print("Result6:", result6)