from genepool import GenePool
from grn import GeneRegulatoryNetwork
from layers import NPointCrossover, UniformMutation
from environments import Environment
from selection import RandomSelection
import numpy as np


def phenotype_fitness_function(phenotypes):
    return_values = []
    for phenotype in phenotypes:
        # Convert the tensor to a NumPy array before summing
        return_values.append(np.sum(phenotype.detach().numpy()))
    return return_values


cross_selection = RandomSelection(amount_to_select=2)
insertion_selection = RandomSelection(amount_to_select=2)

gene_regulatory_network = GeneRegulatoryNetwork(input_length=5, hidden_size=5, num_layers=3, output_shape=(4,))
gene_pool = GenePool(size=5, grn=gene_regulatory_network)

environment = Environment(
    layers = [
        NPointCrossover(selection_function=cross_selection.select, families=2, children=2, n_points=3),
        UniformMutation(selection_function=insertion_selection.select, device="cpu", magnitude=0.01)
    ],
    genepool = gene_pool,
    pbf_function = phenotype_fitness_function
)

environment.compile(start_population=20, max_individuals=100)
environment.evolve(generations=100)
environment.plot()