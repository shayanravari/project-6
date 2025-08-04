import torch
import numpy as np

def main():
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a tensor with random values
    tensor = torch.rand(3, 3)
    
    # Print the tensor
    print("Random Tensor:")
    print(tensor)

if __name__ == "__main__":
    main()