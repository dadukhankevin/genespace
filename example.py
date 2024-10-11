# example.py
from genepool import GenePool
from grn import GeneRegulatoryNetwork
from genespace.layers import NPointCrossover, UniformMutation
from genespace.environments import Environment
from genespace.selection import RandomSelection
import numpy as np

def phenotype_fitness_function(phenotypes):
    return_values = []
    for phenotype in phenotypes:
        # Convert the tensor to a NumPy array before summing
        return_values.append(np.sum(phenotype.cpu().numpy()))
    return return_values

cross_selection = RandomSelection(amount_to_select=lambda: 2)
insertion_selection = RandomSelection(amount_to_select=lambda: 2)

# Initialize GRN with desired input and output dimensions
input_length = 5
output_shape = (10,)
gene_regulatory_network = GeneRegulatoryNetwork(input_length=input_length, hidden_size=5, num_layers=3, output_shape=output_shape, device='cpu')

gene_pool = GenePool(size=input_length, grn=gene_regulatory_network)  # size is the input_length

environment = Environment(
    layers = [
        NPointCrossover(selection_function=cross_selection.select, families=4, children=2, n_points=3),
        UniformMutation(selection_function=insertion_selection.select, device="cpu", magnitude=0.01)
    ],
    genepool = gene_pool,
    pbf_function = phenotype_fitness_function
)

environment.compile(start_population=20, max_individuals=100)
environment.evolve(generations=100, backprop_mode='divide_and_conquer')  # Choose mode as needed
environment.plot()
