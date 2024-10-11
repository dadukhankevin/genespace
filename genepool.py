import torch
import numpy as np
from individual import Individual
from grn import GeneRegulatoryNetwork


class GenePool:
    def __init__(self, size, grn: GeneRegulatoryNetwork, binary_mode=False):
        self.size = size
        self.grn = grn
        self.binary_mode = binary_mode

    def create_genes(self):
        if self.binary_mode:
            return np.random.randint(0, 2, self.size).astype(np.float32)
        else:
            return np.random.randn(self.size).astype(np.float32)
    
    def generate_one_gene(self):
        if self.binary_mode:
            return np.random.randint(0, 2, 1).astype(np.float32)
        else:
            return np.random.randn(1).astype(np.float32)
    
    def create_individual(self):
        genes = self.create_genes()
        return Individual(genes=genes, grn=self.grn)

if __name__ == "__main__":
    pool = GenePool(100, GeneRegulatoryNetwork(100))
    print(pool.create_individual())
