import torch
import numpy as np
from grn import GeneRegulatoryNetwork

class Individual:
    def __init__(self, genes: np.ndarray, grn: GeneRegulatoryNetwork):
        self.grn = grn
        self.genes = genes  # Now a NumPy array
        self.fitness = 0
        self.modified = True

    def __call__(self):
        genes_tensor = torch.from_numpy(self.genes).float().to(self.grn.device)  # Convert to tensor when needed
        return self.grn.forward(genes_tensor)


