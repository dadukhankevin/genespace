# decoders.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GeneSpaceDecoderBase(nn.Module):
    def __init__(self, input_length, output_shape=(10,), lr=0.001, device='cpu'):
        super(GeneSpaceDecoderBase, self).__init__()
        
        self.input_length = input_length
        self.output_shape = output_shape
        self.device = device
        self.lr = lr
        
        # Calculate total output size
        self.output_size = torch.prod(torch.tensor(output_shape)).item()
        
        # Placeholder for the model, to be defined in subclasses
        self.model = None
        
        self.to(self.device)  # Ensure the model is on the specified device
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
    
    def initialize_optimizer(self):
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def backprop_network(self, individuals, train_top_percent=0.5, batch_size=32):
        """
        Trains the decoder using the genes of less fit individuals as input
        and the phenotypes of top individuals as target.
        
        Parameters:
        - individuals: List of Individuals in the current population.
        - train_top_percent (float): Percentage (0.0 to 1.0) of top individuals to use for target phenotypes.
        - batch_size (int): Number of individuals to process in each batch.
        """
        self.train()
        
        # Sort individuals based on fitness (descending order)
        sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
        population_size = len(sorted_individuals)
        
        if population_size < 2:
            print("Not enough individuals available for training.")
            return
        
        # Determine the split point for top and bottom individuals
        split_index = max(1, int(population_size * train_top_percent))
        
        # Prepare training data
        top_individuals = sorted_individuals[:split_index]
        bottom_individuals = sorted_individuals[split_index:]
        
        # X (genes of less fit individuals)
        X = torch.tensor([ind.genes for ind in bottom_individuals], dtype=torch.float32).to(self.device)
        
        total_loss = 0
        num_batches = (len(X) + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(num_batches):
            self.optimizer.zero_grad()
            
            # Get batch for X
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch_X = X[start_idx:end_idx]
            
            # Get batch for Y (phenotypes of top individuals)
            start_idx_Y = (i * batch_size) % len(top_individuals)
            end_idx_Y = min(start_idx_Y + batch_size, len(top_individuals))
            batch_top_individuals = top_individuals[start_idx_Y:end_idx_Y]
            
            # Forward pass for Y
            batch_Y = torch.cat([self.forward(torch.tensor(ind.genes, dtype=torch.float32).unsqueeze(0).to(self.device)) for ind in batch_top_individuals])
            
            # Forward pass for X
            predictions = self.forward(batch_X)
            
            # Ensure predictions and batch_Y have the same size
            min_size = min(predictions.size(0), batch_Y.size(0))
            predictions = predictions[:min_size]
            batch_Y = batch_Y[:min_size]
            
            # Compute loss
            loss = self.criterion(predictions, batch_Y)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches

class MLPGeneSpaceDecoder(GeneSpaceDecoderBase):
    def __init__(self, input_length, hidden_size=64, num_layers=3, output_shape=(10,), lr=0.001, device='cpu', activation=nn.LeakyReLU, output_activation=nn.Sigmoid):
        super(MLPGeneSpaceDecoder, self).__init__(input_length, output_shape, lr, device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input layer
        layers = [nn.Linear(input_length, hidden_size), activation()]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation()])
        
        # Output layer
        layers.append(nn.Linear(hidden_size, self.output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Activation function for the output
        self.output_activation = output_activation()
        
        self.to(self.device)  # Ensure the model is on the specified device
        self.initialize_optimizer()  # Initialize the optimizer after the model is defined
    
    def forward(self, x):
        # x shape: (batch_size, input_length)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x).float()
        
        x = x.to(self.device)
        
        # Process through MLP
        output = self.model(x)
        
        # Apply activation and reshape to desired output shape
        output = self.output_activation(output)
        output = output.view(-1, *self.output_shape)
        
        return output

class GRUGeneSpaceDecoder(GeneSpaceDecoderBase):
    def __init__(self, input_length, hidden_size=64, num_layers=1, output_shape=(10,), lr=0.001, device='cpu', output_activation=nn.Sigmoid):
        super(GRUGeneSpaceDecoder, self).__init__(input_length, output_shape, lr, device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define GRU layer
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, self.output_size)
        
        # Activation function for the output
        self.output_activation = output_activation()
        
        self.to(self.device)  # Ensure the model is on the specified device
        self.initialize_optimizer()  # Initialize the optimizer after the model is defined
    
    def forward(self, x):
        # x shape: (batch_size, input_length)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x).float()
        
        x = x.to(self.device)
        
        # Reshape x to (batch_size, sequence_length, input_size)
        x = x.unsqueeze(-1)  # Now x shape: (batch_size, input_length, 1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Process through GRU
        out, _ = self.gru(x, h0)  # out shape: (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step
        out = out[:, -1, :]  # out shape: (batch_size, hidden_size)
        
        # Output layer
        out = self.fc(out)
        
        # Apply activation and reshape to desired output shape
        out = self.output_activation(out)
        out = out.view(-1, *self.output_shape)
        
        return out

class ConvolutionalGeneSpaceDecoder(GeneSpaceDecoderBase):
    def __init__(self, input_length, output_shape=(1, 28, 28), lr=0.001, device='cpu', activation=nn.LeakyReLU, output_activation=nn.Sigmoid):
        super(ConvolutionalGeneSpaceDecoder, self).__init__(input_length, output_shape, lr, device)
        
        self.activation = activation()
        self.output_activation = output_activation()
        
        # Define the model
        # Since we're mapping a flat genotype to an image, we'll use fully connected layers followed by reshaping
        # For actual convolutional layers, more complex architectures are needed
        self.model = nn.Sequential(
            nn.Linear(input_length, 256),
            self.activation,
            nn.Linear(256, 512),
            self.activation,
            nn.Linear(512, 1024),
            self.activation,
            nn.Linear(1024, self.output_size),
            self.output_activation
        )
        
        self.to(self.device)
        self.initialize_optimizer()  # Initialize the optimizer after the model is defined
    
    def forward(self, x):
        # x shape: (batch_size, input_length)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x).float()
        
        x = x.to(self.device)
        
        # Forward pass
        output = self.model(x)
        
        # Reshape to desired output shape
        output = output.view(-1, *self.output_shape)
        
        return output
