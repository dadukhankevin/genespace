# environments.py
from genespace.layers import Layer
from genespace.individual import Individual
from genespace.genepool import GenePool
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import torch

class Environment:
    def __init__(self, layers: list[Layer], genepool: GenePool, pbf_function: Callable):
        self.layers: list[Layer] = layers
        self.individuals: list[Individual] = []
        self.max_individuals: int = 0
        self.compiled = False
        self.start_population: int = 2
        self.early_stop: float = float('inf')
        self.batch_size: int = 10
        self.genepool: GenePool = genepool
        self.pbf_function: Callable = pbf_function
        self.fitness_history = []
        self.population_history = []
    
    def compile(self, start_population: int, max_individuals: int, batch_size: int = 10, individuals: list[Individual] = [], early_stop: float = float('inf')):
        assert start_population >= 2, "Start population must be at least 2"

        self.individuals = individuals
        self.start_population = start_population
        self.max_individuals = max_individuals
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.compiled = True

        for layer in self.layers:
            layer.initialize(self)
    
    def batch_fitness(self):
        individuals_for_measurement = [individual for individual in self.individuals if individual.modified]
        
        if not individuals_for_measurement:
            return
        
        batch_size = self.batch_size
        for i in range(0, len(individuals_for_measurement), batch_size):
            batch = individuals_for_measurement[i:i+batch_size]
            batch_genes = torch.tensor([ind.genes for ind in batch], dtype=torch.float32).to(self.genepool.grn.device)
            
            # Debug print

            
            phenotypes = self.genepool.grn.forward(batch_genes).detach()
            batch_fitnesses = self.pbf_function(phenotypes)
            
            for individual, fitness in zip(batch, batch_fitnesses):
                individual.fitness = fitness
                individual.modified = False

    def sort_individuals(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
    
    def evolve(self, generations=100, backprop_mode='divide_and_conquer', backprop_every_n=1, epochs=1):
        assert self.compiled, "Environment must be compiled before evolving"

        for i in range(generations):
            for layer in self.layers:
                while len(self.individuals) < self.start_population:
                    self.individuals.append(self.genepool.create_individual())
                new_individuals = layer.execute()
                if new_individuals:
                    self.individuals.extend(new_individuals)
                    self.batch_fitness()
                    self.sort_individuals()
                    self.individuals = self.individuals[:self.max_individuals]
                if self.individuals and self.individuals[0].fitness > self.early_stop:
                    print(f"Early stopping at generation {i} with fitness {self.individuals[0].fitness}")
                    return self.individuals
            
            # Train the GRN every backprop_every_n generations
            if i % backprop_every_n == 0:
                for _ in range(epochs):
                    loss = self.genepool.grn.backprop_network(self.individuals, mode=backprop_mode)
            
            self.fitness_history.append(self.individuals[0].fitness)
            self.population_history.append(len(self.individuals))
            
            print(f"Generation: {i}")
            print("Max fitness: ", self.individuals[0].fitness)
            print("Population size: ", len(self.individuals))
            if i % backprop_every_n == 0:
                print(f"GRN Training Loss: {loss}\n")
            else:
                print()
    
        return self.individuals
    
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot fitness history
        fitness_history = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in self.fitness_history]
        ax1.plot(fitness_history)
        ax1.set_title('Fitness History')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Max Fitness')

        # Plot population history
        population_history = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in self.population_history]
        ax2.plot(population_history)
        ax2.set_title('Population History')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population Size')

        plt.tight_layout()
        plt.show()
