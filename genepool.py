import torch
import numpy as np
from genespace.individual import Individual
from genespace.decoders import MLPGeneSpaceDecoder


class GenePool:
    def __init__(self, size, gsp: MLPGeneSpaceDecoder, binary_mode=False):
        self.size = size
        self.gsp = gsp
        self.binary_mode = binary_mode

    def create_genes(self):
        if self.binary_mode:
            return np.random.randint(0, 2, self.size).astype(np.int8)
        else:
            return np.random.randn(self.size).astype(np.float32)
    
    def generate_one_gene(self):
        if self.binary_mode:
            return np.random.randint(0, 2, 1).astype(np.int8)
        else:
            return np.random.randn(1).astype(np.float32)
    
    def create_individual(self):
        genes = self.create_genes()
        return Individual(genes=genes, gsp=self.gsp)

if __name__ == "__main__":
    pool = GenePool(100, MLPGeneSpaceDecoder(100))
    print(pool.create_individual())
