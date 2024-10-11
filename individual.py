import torch
import numpy as np
from genespace.grn import GeneRegulatoryNetwork

class Individual:
    def __init__(self, genes: np.ndarray, grn: GeneRegulatoryNetwork):
        self.grn = grn
        self.genes = genes  # Now a NumPy array
        self.fitness = 0
        self.modified = True
