# decoders.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import enum

class TrainingMode(enum.Enum):
    GOOD_TO_BEST = 1 # X = genotypes of the top % of individuals, Y = phenotype (output) of the best individual.
    EACH_TO_NEXT = 2 # X = random genotypes from the individuals (by %), Y = the phenotypes of each next individual.
    TOP_AND_BOTTOM_PERCENT = 3 # X = genotypes of the bottom % of individuals, Y = phenotypes of the top % of individuals (what is currently implimented)

class GeneSpaceDecoderBase(nn.Module):
    def __init__(self, input_length, output_shape=(10,), lr=0.001, device='cpu', training_mode=TrainingMode.TOP_AND_BOTTOM_PERCENT):
        super(GeneSpaceDecoderBase, self).__init__()
        
        self.input_length = input_length
        self.output_shape = output_shape
        self.device = device
        self.lr = lr
        
        # Calculate total output size
        self.output_size = torch.prod(torch.tensor(output_shape)).item()
        
        # Placeholder for the model, to be defined in subclasses
        self.model = None
        
        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Add TrainingMode
        self.training_mode = training_mode
    
    def initialize_optimizer(self):
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def backprop_network(self, individuals, selection_percent=0.5, batch_size=32):
        if self.training_mode == TrainingMode.TOP_AND_BOTTOM_PERCENT:
            return self._backprop_top_and_bottom(individuals, selection_percent, batch_size)
        elif self.training_mode == TrainingMode.GOOD_TO_BEST:
            return self._backprop_good_to_best(individuals, selection_percent, batch_size)
        elif self.training_mode == TrainingMode.EACH_TO_NEXT:
            return self._backprop_each_to_next(individuals, selection_percent, batch_size)
        else:
            raise ValueError("Invalid training mode")

    def _backprop_top_and_bottom(self, individuals, selection_percent, batch_size):
        # Sort individuals by fitness
        amount = int(len(individuals) * selection_percent)
        assert amount + amount <= len(individuals) and selection_percent < 0.5, "Selection percent is too high"

        top_individuals = individuals[:amount]
        bottom_individuals = individuals[-amount:]
        top_genes = [ind.genes for ind in top_individuals]
        bottom_genes = [ind.genes for ind in bottom_individuals]
        assert len(top_individuals) == len(bottom_individuals), "Top and bottom individuals must have the same length"

        top_batches = [top_genes[i:i+batch_size] for i in range(0, len(top_genes), batch_size)]
        bottom_batches = [bottom_genes[i:i+batch_size] for i in range(0, len(bottom_genes), batch_size)]
        
        top_phenotypes = []
        for top_batch in top_batches:
            top_phenotypes.append(self.forward(torch.tensor(top_batch, dtype=torch.float32, device=self.device)))
        
        # Now do a backward pass of bottom_genes to top_phenotypes
        total_loss = 0
        for bottom_batch, top_phenotype in zip(bottom_batches, top_phenotypes):
            bottom_batch_tensor = torch.tensor(bottom_batch, dtype=torch.float32, device=self.device)
            target_phenotype = top_phenotype.detach()  # Detach to prevent gradients flowing through targets
            
            self.optimizer.zero_grad()
            predicted_phenotype = self.forward(bottom_batch_tensor)
            loss = self.criterion(predicted_phenotype, target_phenotype)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(bottom_batches)
        return average_loss

    def _backprop_good_to_best(self, individuals, selection_percent, batch_size):
        # Select the top individuals
        num_selected = int(len(individuals) * selection_percent)
        selected_individuals = individuals[:num_selected]
        
        # Get the best individual
        best_individual = individuals[0]
        
        # Prepare batches
        genes = [ind.genes for ind in selected_individuals]
        batches = [genes[i:i+batch_size] for i in range(0, len(genes), batch_size)]
        
        total_loss = 0
        for batch in batches:
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
            target_phenotype = self.forward(torch.tensor(best_individual.genes, dtype=torch.float32, device=self.device)).detach()
            
            self.optimizer.zero_grad()
            predicted_phenotype = self.forward(batch_tensor)
            loss = self.criterion(predicted_phenotype, target_phenotype.expand_as(predicted_phenotype))
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(batches)
        return average_loss

    def _backprop_each_to_next(self, individuals, selection_percent, batch_size):
        # Select the top individuals
        num_selected = int(len(individuals) * selection_percent)
        selected_individuals = individuals[:num_selected]
        
        total_loss = 0
        num_individuals = len(selected_individuals) - 1  # -1 because we're pairing each with the next

        for i in range(0, num_individuals, batch_size):
            batch = selected_individuals[i:i+batch_size]
            next_individuals = selected_individuals[i+1:i+batch_size+1]
            
            batch_genes = torch.tensor([ind.genes for ind in batch], dtype=torch.float32, device=self.device)
            next_genes = torch.tensor([ind.genes for ind in next_individuals], dtype=torch.float32, device=self.device)
            
            self.optimizer.zero_grad()
            predicted_phenotypes = self.forward(batch_genes)
            target_phenotypes = self.forward(next_genes).detach()
            
            # Compute loss for each individual-next pair
            for j in range(len(batch)):
                loss = self.criterion(predicted_phenotypes[j].unsqueeze(0), target_phenotypes[j].unsqueeze(0))
                loss.backward(retain_graph=True)
                total_loss += loss.item()
            
            self.optimizer.step()

        average_loss = total_loss / num_individuals
        return average_loss

    def apply_random_gradient(self, individuals, n_gradients, pbf_function, selection_percent=0.5, batch_size=32):
        # Sample generate n_gradients to apply to the network
        gradients = []
        for _ in range(n_gradients):
            gradient = {}
            for name, param in self.named_parameters():
                gradient[name] = torch.randn_like(param, device=self.device) * self.lr
            gradients.append(gradient)

        # Select the correct amount of individuals using the selection_percent
        num_selected = int(len(individuals) * selection_percent)
        selected_individuals = individuals[:num_selected]

        # Generate phenotypes once, outside the loop
        genes = torch.tensor([ind.genes for ind in selected_individuals], dtype=torch.float32, device=self.device)

        # Test each gradient
        original_state_dict = self.state_dict()
        best_gradient = None
        best_fitness = float('-inf')

        for gradient in gradients:
            # Apply gradient
            for name, param in self.named_parameters():
                param.data += gradient[name]

            # Generate phenotypes in batches
            phenotypes = []
            for i in range(0, len(genes), batch_size):
                batch = genes[i:i+batch_size]
                phenotypes.append(self.forward(batch))
            phenotypes = torch.cat(phenotypes, dim=0)

            # Test fitness
            fitnesses = pbf_function(phenotypes)
            avg_fitness = np.mean(fitnesses)

            if avg_fitness > best_fitness:
                best_fitness = avg_fitness
                best_gradient = gradient

            # Restore original network state
            self.load_state_dict(original_state_dict)

        # Apply the best gradient if it improves fitness
        if best_gradient is not None:
            current_fitness = np.mean(pbf_function(self.forward(genes)))
            if best_fitness >= current_fitness:
                for name, param in self.named_parameters():
                    param.data += best_gradient[name]
                return best_fitness
        
        return None  # Return None if no gradient improved fitness

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
