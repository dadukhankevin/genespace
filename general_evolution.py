# Experimental 
from genespace import layers, genepool, decoders, selection, environments
import enum

class GeneMode(enum.Enum):
    BINARY = 1  # Binary genes
    REAL = 2  # Real-valued genes

class GeneralEvolution:
    def __init__(self, input_length, hidden_size, num_layers, output_shape, gene_mode=GeneMode.BINARY):


        self.decoder = decoders.MLPGeneSpaceDecoder(input_length=input_length, hidden_size=hidden_size, num_layers=num_layers, output_shape=output_shape)
        self.genepool = genepool.GenePool(size=input_length, gsp=self.decoder, binary=gene_mode == GeneMode.BINARY)

        self.environment = environments.Environment(layers=[
            layers.NPointCrossover(selection_function=selection.RandomSelection(amount_to_select=lambda: 2), families=4, children=2, n_points=3),
            layers.UniformMutation(selection_function=selection.RandomSelection(amount_to_select=lambda: 2), device="cpu", magnitude=0.01) if gene_mode == GeneMode.BINARY else layers.BinaryFlipMutation(selection_function=selection.RandomSelection(amount_to_select=lambda: 2), device="cpu", magnitude=0.01)
        ], genepool=self.genepool, pbf_function=self.decoder.forward)

    def evolve(self, generations=100, backprop_mode=environments.BackpropMode.GRADIENT_DESCENT, backprop_every_n=1, epochs=1, selection_percent=.5, batch_size=32):
        self.environment.compile(start_population=20, max_individuals=100)
        self.environment.evolve(generations=generations, backprop_mode=backprop_mode, backprop_every_n=backprop_every_n, epochs=epochs, selection_percent=selection_percent, batch_size=batch_size)