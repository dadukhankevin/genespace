import torch
import numpy as np
from genespace.decoders import MLPGeneSpaceDecoder

class Individual:
    def __init__(self, genes: np.ndarray, gsp: MLPGeneSpaceDecoder):
        self.gsp = gsp 
        self.genes = genes  # Now a NumPy array
        self.fitness = 0
        self.modified = True
