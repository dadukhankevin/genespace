import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GeneRegulatoryNetwork(nn.Module):
    def __init__(self, input_length, hidden_size=64, num_layers=3, output_shape=(10, 5, 3), device='cpu'):
        super(GeneRegulatoryNetwork, self).__init__()
        
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_shape = output_shape
        self.device = device
        
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
        
        self.to(self.device)  # Ensure the model is on the specified device
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        # x shape: (batch_size, input_length)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x).float()
        
        x = x.to(self.device)
        
        # Process through MLP
        output = self.mlp(x)
        
        # Apply activation and reshape to desired output shape
        output = self.output_activation(output)
        output = output.view(-1, *self.output_shape)
        
        return output
    
    def backprop_network(self, individuals, mode='divide_and_conquer'):
        """
        Trains the GRN based on the specified mode.
        
        Parameters:
        - individuals: List of Individuals in the current population.
        - mode (str): The training strategy. Options: 'divide_and_conquer', 'weighted', 'direct_prediction'
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Sort individuals based on fitness
        sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
        population_size = len(sorted_individuals)
        
        if population_size < 2:
            print("Not enough individuals to train the GRN.")
            return
        
        if mode == 'divide_and_conquer':
            # Split population into bottom 50% and top 50%
            midpoint = population_size // 2
            bottom_half = sorted_individuals[midpoint:]
            top_half = sorted_individuals[:midpoint]
            
            # Prepare training data
            bottom_genes = torch.tensor([ind.genes for ind in bottom_half], dtype=torch.float32).to(self.device)
            top_genes = torch.tensor([ind.genes for ind in top_half], dtype=torch.float32).to(self.device)
            
            # Compute phenotypes for top individuals (detach to prevent gradients)
            with torch.no_grad():
                top_phenotypes = self.forward(top_genes)
            top_phenotypes = top_phenotypes.view(len(top_half), -1)
            
            # Forward pass on bottom genes
            predictions = self.forward(bottom_genes)
            predictions = predictions.view(len(bottom_half), -1)
            
            # Compute loss
            loss = self.criterion(predictions, top_phenotypes)
        
        elif mode == 'weighted':
            # Assign weights based on fitness rank
            weights = torch.tensor([ind.fitness for ind in sorted_individuals], dtype=torch.float32).to(self.device)
            weights = weights / weights.sum()
            
            genes = torch.tensor([ind.genes for ind in sorted_individuals], dtype=torch.float32).to(self.device)
            
            # Compute phenotypes (detach to prevent gradients)
            with torch.no_grad():
                phenotypes = self.forward(genes)
            phenotypes = phenotypes.view(population_size, -1)
            
            # Forward pass
            predictions = self.forward(genes)
            predictions = predictions.view(population_size, -1)
            
            # Compute weighted loss
            losses = self.criterion(predictions, phenotypes, reduction='none')
            losses = losses.mean(dim=1)  # Mean over output dimensions
            loss = (losses * weights).sum()
        
        elif mode == 'direct_prediction':
            # Train on top individuals to predict their phenotypes from their genes
            top_half = sorted_individuals[:population_size // 2]
            
            genes = torch.tensor([ind.genes for ind in top_half], dtype=torch.float32).to(self.device)
            
            # Compute phenotypes (detach to prevent gradients)
            with torch.no_grad():
                phenotypes = self.forward(genes)
            phenotypes = phenotypes.view(len(top_half), -1)
            
            # Forward pass
            predictions = self.forward(genes)
            predictions = predictions.view(len(top_half), -1)
            
            # Compute loss
            loss = self.criterion(predictions, phenotypes)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


        

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
