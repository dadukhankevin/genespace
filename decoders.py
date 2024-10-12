# train_top_percent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GeneSpaceDecoderBase(nn.Module):
    def __init__(self, input_length, output_shape=(10, 5, 3), lr=0.001, device='cpu'):
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
        # Initialize optimizer after the model parameters are defined
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def backprop_network(self, individuals, mode='divide_and_conquer', train_top_percent=1.0):
        """
        Trains the decoder based on the specified mode.
        
        Parameters:
        - individuals: List of Individuals in the current population.
        - mode (str): The training strategy. Options: 'divide_and_conquer', 'direct_target'
        - train_top_percent (float): Percentage (0.0 to 1.0) of top individuals to use for training.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def set_target_image(self, target_image):
        """
        Sets the target image for training in 'direct_target' mode.
        
        Parameters:
        - target_image: A tensor or NumPy array representing the target image.
        """
        if isinstance(target_image, np.ndarray):
            target_image = torch.from_numpy(target_image).float()
        self.target_image = target_image.to(self.device)

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
        
        # Initialize optimizer after model parameters are defined
        self.initialize_optimizer()
    
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
    
    def backprop_network(self, individuals, mode='divide_and_conquer', train_top_percent=1.0):
        """
        Trains the decoder based on the specified mode.
        
        Parameters:
        - individuals: List of Individuals in the current population.
        - mode (str): The training strategy. Options: 'divide_and_conquer', 'direct_target'
        - train_top_percent (float): Percentage (0.0 to 1.0) of top individuals to use for training.
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Sort individuals based on fitness
        sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
        population_size = len(sorted_individuals)
        
        if population_size < 2:
            print("Not enough individuals to train the decoder.")
            return
        
        # Determine the number of top individuals to use
        num_top_individuals = max(1, int(population_size * train_top_percent))
        top_individuals = sorted_individuals[:num_top_individuals]
        
        if mode == 'divide_and_conquer':
            # Split population into bottom and top individuals
            bottom_individuals = sorted_individuals[num_top_individuals:]
            
            if len(bottom_individuals) == 0:
                print("No bottom individuals to train on.")
                return
            
            # Prepare training data
            bottom_genes = torch.tensor([ind.genes for ind in bottom_individuals], dtype=torch.float32).to(self.device)
            top_genes = torch.tensor([ind.genes for ind in top_individuals], dtype=torch.float32).to(self.device)
            
            # Compute phenotypes for top individuals (detach to prevent gradients)
            with torch.no_grad():
                top_phenotypes = self.forward(top_genes)
            top_phenotypes = top_phenotypes.view(len(top_individuals), -1)
            
            # Forward pass on bottom genes
            predictions = self.forward(bottom_genes)
            predictions = predictions.view(len(bottom_individuals), -1)
            
            # Repeat top phenotypes to match bottom batch size
            repeated_top_phenotypes = top_phenotypes.repeat(len(bottom_individuals) // len(top_individuals) + 1, 1)
            repeated_top_phenotypes = repeated_top_phenotypes[:len(bottom_individuals)]
            
            # Compute loss
            loss = self.criterion(predictions, repeated_top_phenotypes)
        
        elif mode == 'direct_target':
            if not hasattr(self, 'target_image'):
                raise ValueError("Target image not set. Please use set_target_image() method.")
            
            # Use top individuals for training
            genotypes = torch.tensor([ind.genes for ind in top_individuals], dtype=torch.float32).to(self.device)
            batch_size = genotypes.size(0)
            
            # Forward pass
            outputs = self.forward(genotypes)
            outputs = outputs.view(batch_size, -1)
            
            # Prepare target images
            target_images = self.target_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            target_images = target_images.view(batch_size, -1)
            
            # Compute loss
            loss = self.criterion(outputs, target_images)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

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
        
        # Initialize optimizer after model parameters are defined
        self.initialize_optimizer()
    
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
    
    def backprop_network(self, individuals, mode='divide_and_conquer', train_top_percent=1.0):
        """
        Trains the decoder based on the specified mode.
        
        Parameters:
        - individuals: List of Individuals in the current population.
        - mode (str): The training strategy. Options: 'divide_and_conquer', 'direct_target'
        - train_top_percent (float): Percentage (0.0 to 1.0) of top individuals to use for training.
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Sort individuals based on fitness
        sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
        population_size = len(sorted_individuals)
        
        if population_size < 2:
            print("Not enough individuals to train the decoder.")
            return
        
        # Determine the number of top individuals to use
        num_top_individuals = max(1, int(population_size * train_top_percent))
        top_individuals = sorted_individuals[:num_top_individuals]
        
        if mode == 'divide_and_conquer':
            # Split population into bottom and top individuals
            bottom_individuals = sorted_individuals[num_top_individuals:]
            
            if len(bottom_individuals) == 0:
                print("No bottom individuals to train on.")
                return
            
            # Prepare training data
            bottom_genes = torch.tensor([ind.genes for ind in bottom_individuals], dtype=torch.float32).to(self.device)
            top_genes = torch.tensor([ind.genes for ind in top_individuals], dtype=torch.float32).to(self.device)
            
            # Compute phenotypes for top individuals (detach to prevent gradients)
            with torch.no_grad():
                top_phenotypes = self.forward(top_genes)
            top_phenotypes = top_phenotypes.view(len(top_individuals), -1)
            
            # Forward pass on bottom genes
            predictions = self.forward(bottom_genes)
            predictions = predictions.view(len(bottom_individuals), -1)
            
            # Repeat top phenotypes to match bottom batch size
            repeated_top_phenotypes = top_phenotypes.repeat(len(bottom_individuals) // len(top_individuals) + 1, 1)
            repeated_top_phenotypes = repeated_top_phenotypes[:len(bottom_individuals)]
            
            # Compute loss
            loss = self.criterion(predictions, repeated_top_phenotypes)
        
        elif mode == 'direct_target':
            if not hasattr(self, 'target_image'):
                raise ValueError("Target image not set. Please use set_target_image() method.")
            
            # Use top individuals for training
            genotypes = torch.tensor([ind.genes for ind in top_individuals], dtype=torch.float32).to(self.device)
            batch_size = genotypes.size(0)
            
            # Forward pass
            outputs = self.forward(genotypes)
            outputs = outputs.view(batch_size, -1)
            
            # Prepare target images
            target_images = self.target_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            target_images = target_images.view(batch_size, -1)
            
            # Compute loss
            loss = self.criterion(outputs, target_images)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class ConvolutionalGeneSpaceDecoder(GeneSpaceDecoderBase):
    def __init__(self, input_length, output_shape=(1, 28, 28), lr=0.001, device='cpu', activation=nn.LeakyReLU, output_activation=nn.Sigmoid):
        super(ConvolutionalGeneSpaceDecoder, self).__init__(input_length, output_shape, lr, device)
        
        self.activation = activation()
        self.output_activation = output_activation()
        
        # Define the model
        # Start with Linear layers to map input_length to a feature map suitable for convolution
        # For example, map to 256 features, then reshape to (batch_size, 16, 4, 4) for convolutions
        self.model = nn.Sequential(
            nn.Linear(input_length, 256),
            self.activation,
            nn.Linear(256, 512),
            self.activation,
            nn.Linear(512, 1024),
            self.activation,
            nn.Linear(1024, output_shape[0] * output_shape[1] * output_shape[2]),
            self.output_activation
        )
        
        self.to(self.device)
        
        # Initialize optimizer after model parameters are defined
        self.initialize_optimizer()
    
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
    
    def backprop_network(self, individuals, mode='divide_and_conquer', train_top_percent=1.0):
        """
        Trains the decoder based on the specified mode.
        
        Parameters:
        - individuals: List of Individuals in the current population.
        - mode (str): The training strategy. Options: 'divide_and_conquer', 'direct_target'
        - train_top_percent (float): Percentage (0.0 to 1.0) of top individuals to use for training.
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Sort individuals based on fitness
        sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
        population_size = len(sorted_individuals)
        
        if population_size < 2:
            print("Not enough individuals to train the decoder.")
            return
        
        # Determine the number of top individuals to use
        num_top_individuals = max(1, int(population_size * train_top_percent))
        top_individuals = sorted_individuals[:num_top_individuals]
        
        if mode == 'divide_and_conquer':
            # Split population into bottom and top individuals
            bottom_individuals = sorted_individuals[num_top_individuals:]
            
            if len(bottom_individuals) == 0:
                print("No bottom individuals to train on.")
                return
            
            # Prepare training data
            bottom_genes = torch.tensor([ind.genes for ind in bottom_individuals], dtype=torch.float32).to(self.device)
            top_genes = torch.tensor([ind.genes for ind in top_individuals], dtype=torch.float32).to(self.device)
            
            # Compute phenotypes for top individuals (detach to prevent gradients)
            with torch.no_grad():
                top_phenotypes = self.forward(top_genes)
            top_phenotypes = top_phenotypes.view(len(top_individuals), -1)
            
            # Forward pass on bottom genes
            predictions = self.forward(bottom_genes)
            predictions = predictions.view(len(bottom_individuals), -1)
            
            # Repeat top phenotypes to match bottom batch size
            repeated_top_phenotypes = top_phenotypes.repeat(len(bottom_individuals) // len(top_individuals) + 1, 1)
            repeated_top_phenotypes = repeated_top_phenotypes[:len(bottom_individuals)]
            
            # Compute loss
            loss = self.criterion(predictions, repeated_top_phenotypes)
        
        elif mode == 'direct_target':
            if not hasattr(self, 'target_image'):
                raise ValueError("Target image not set. Please use set_target_image() method.")
            
            # Use top individuals for training
            genotypes = torch.tensor([ind.genes for ind in top_individuals], dtype=torch.float32).to(self.device)
            batch_size = genotypes.size(0)
            
            # Forward pass
            outputs = self.forward(genotypes)
            outputs = outputs.view(batch_size, -1)
            
            # Prepare target images
            target_images = self.target_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            target_images = target_images.view(batch_size, -1)
            
            # Compute loss
            loss = self.criterion(outputs, target_images)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    # Test section
if __name__ == "__main__":
    import torch

    # Test parameters
    input_length = 10
    output_shape = (3, 32, 32)  # Example output shape
    batch_size = 5

    # Create dummy input data
    dummy_input = torch.randn(batch_size, input_length)

    # Test MLPGeneSpaceDecoder
    print("Testing MLPGeneSpaceDecoder...")
    mlp_decoder = MLPGeneSpaceDecoder(input_length, output_shape=output_shape)
    mlp_output = mlp_decoder(dummy_input)
    print(f"MLP output shape: {mlp_output.shape}")

    # Test GRUGeneSpaceDecoder
    print("\nTesting GRUGeneSpaceDecoder...")
    gru_decoder = GRUGeneSpaceDecoder(input_length, output_shape=output_shape)
    gru_output = gru_decoder(dummy_input)
    print(f"GRU output shape: {gru_output.shape}")

    # Test ConvolutionalGeneSpaceDecoder
    print("\nTesting ConvolutionalGeneSpaceDecoder...")
    conv_decoder = ConvolutionalGeneSpaceDecoder(input_length, output_shape=output_shape)
    conv_output = conv_decoder(dummy_input)
    print(f"Convolutional output shape: {conv_output.shape}")

    print("\nAll tests completed successfully!")