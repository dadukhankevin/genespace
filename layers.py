from genespace.genepool import GenePool
from genespace.individual import Individual  
import numpy as np
from typing import Callable
import random

class Layer:
    def __init__(self):
        self.genepool = None
        self.env = None

    def initialize(self, env):
        self.env = env
        self.genepool = env.genepool

    def execute(self):
        pass

    def mark_modified(self, individual):
        individual.modified = True

class NPointCrossover(Layer):
    def __init__(self, selection_function: Callable, families: int, children: int, n_points: int = 3, device='cpu'):
        super().__init__()  # Placeholder GenePool, will be set in initialize()
        self.selection_function = selection_function
        self.families = families
        self.children = children
        self.n_points = n_points
        self.device = device


    def parent(self, parents):
        parent1, parent2 = parents
        children = []

        for _ in range(self.children):
            # Generate n random crossover points
            if self.device == "cpu":
                crossover_points = sorted(np.random.permutation(len(parent1.genes))[:self.n_points].tolist())
                child_genes = np.zeros_like(parent1.genes)
            elif self.device == "gpu":
                crossover_points = sorted(
                    np.random.permutation(len(parent1.genes))[:self.n_points].tolist())
                child_genes = np.zeros_like(parent1.genes)

            # Perform n-point crossover
            current_parent = parent1
            start = 0
            for point in crossover_points:
                child_genes[start:point] = current_parent.genes[start:point]
                current_parent = parent2 if current_parent is parent1 else parent1
                start = point

            # Fill in the last segment
            child_genes[start:] = current_parent.genes[start:]

            # Create new individual
            child = Individual(genes=child_genes, gsp=parent1.gsp)
            self.mark_modified(child)

            children.append(child)

        return children

    def execute(self):
        new_individuals = []
        for _ in range(self.families):
            parents = self.selection_function(self.env.individuals)
            assert len(parents) == 2, "Selection function must return two individuals"
            children = self.parent(parents)
            new_individuals.extend(children)
        return new_individuals

    
class UniformMutation(Layer):
    def __init__(self, selection_function: Callable, device: str = 'cpu', magnitude: float = 0.1):
        super().__init__()
        self.selection_function = selection_function
        self.device = device
        self.magnitude = magnitude

    def execute(self):
        new_individuals = []
        parents = self.selection_function(self.env.individuals)
        for parent in parents:
            mutated_individual = self.mutate(parent)
            new_individuals.append(mutated_individual)
        return new_individuals

    def mutate(self, individual: Individual) -> Individual:
        noise = np.random.uniform(-self.magnitude, self.magnitude, size=individual.genes.shape)
        
        mutated_genes = individual.genes + noise

        # Clip values between 0 and 1
        mutated_genes = np.clip(mutated_genes, 0, 1)

        individual.genes = mutated_genes
        self.mark_modified(individual)
        return individual

class BinaryFlipMutation(Layer):
    def __init__(self, selection_function: Callable, flip_rate: float = 0.1):
        super().__init__()
        self.selection_function = selection_function
        self.flip_rate = flip_rate

    def execute(self):
        new_individuals = []
        parents = self.selection_function(self.env.individuals)
        for parent in parents:
            mutated_individual = self.mutate(parent)
            new_individuals.append(mutated_individual)
        return new_individuals

    def mutate(self, individual: Individual) -> Individual:
        flip_mask = np.random.random(individual.genes.shape) < self.flip_rate
        mutated_genes = np.where(flip_mask, 1 - individual.genes, individual.genes)

        individual.genes = mutated_genes
        self.mark_modified(individual)
        return individual

class SwapMutation(Layer):
    def __init__(self, selection_function: Callable, swap_rate: float = 0.1):
        super().__init__()
        self.selection_function = selection_function
        self.swap_rate = swap_rate

    def execute(self):
        new_individuals = []
        parents = self.selection_function(self.env.individuals)
        for parent in parents:
            mutated_individual = self.mutate(parent)
            new_individuals.append(mutated_individual)
        return new_individuals

    def mutate(self, individual: Individual) -> Individual:
        genes = individual.genes.copy()
        num_swaps = int(len(genes) * self.swap_rate)
        
        for _ in range(num_swaps):
            idx1, idx2 = np.random.choice(len(genes), 2, replace=False)
            genes[idx1], genes[idx2] = genes[idx2], genes[idx1]

        individual.genes = genes
        self.mark_modified(individual)
        return individual
