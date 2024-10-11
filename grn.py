import torch
import torch.nn as nn
import numpy as np

class GeneRegulatoryNetwork(nn.Module):
    def __init__(self, input_length, hidden_size=64, num_layers=3, output_shape=(10, 5, 3)):
        super(GeneRegulatoryNetwork, self).__init__()
        
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_shape = output_shape
        
        # Calculate total output size
        self.output_size = torch.prod(torch.tensor(output_shape)).item()
        
        # Input layer
        layers = [nn.Linear(input_length, hidden_size), nn.ReLU()]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_size, self.output_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Activation function for the output
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, input_length)
        
        # Process through MLP
        output = self.mlp(torch.from_numpy(x).float().to(next(self.parameters()).device))
        
        # Apply activation and reshape to desired output shape
        output = self.output_activation(output)
        output = output.view(-1, *self.output_shape)
        
        return output

if __name__ == "__main__":
    # Example usage
    input_length = 1000
    output_shape = (10, 5, 3)  # Example: 10 traits, each with 5 possible values, each with 3 aspects
    batch_size = 2

    grn = GeneRegulatoryNetwork(input_length, output_shape=output_shape)

    # Generate flat list DNA sequences (values between 0 and 1)
    flat_dna = torch.rand((batch_size, input_length))
    print("Flat DNA shape:", flat_dna.shape)
    print("Flat DNA (first 10 values of first sequence):")
    print(flat_dna[0, :10])

    # Forward pass
    phenotypes = grn(flat_dna)
    print("\nPhenotypes shape:", phenotypes.shape)
    print("Phenotypes (first trait of first sequence):")
    print(phenotypes[0, 0])

    # Print model summary
    print("\nModel Summary:")
    print(grn)

    # Print model parameters
    print("\nModel Parameters:")
    for name, param in grn.named_parameters():
        print(f"{name}: {param.shape}")

    # Print total number of parameters
    total_params = sum(p.numel() for p in grn.parameters())
    print(f"\nTotal number of parameters: {total_params}")
