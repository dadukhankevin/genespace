# Experimental 
from genespace import layers, genepool, decoders, selection, environments
import enum


class GeneralEvolution:
    def __init__(self, fitness_function, output_shape: tuple[int, ...], scale: float=1, input_length: int = 250, device: str = "cpu"):

        """
        I have no idea why these values seem to work, but got them through experementation and best guesses.
        I'm sure there's more optimal combinations. 
        Also, adding layers almost always hurts.
        """
        self.scale = scale
        self.hidden_size = int(2000 * (self.scale ** 2))
        print("hidden size: ", self.hidden_size)
        self.learn_rate = 0.00001
        self.max_population = int(200 * self.scale)
        
        self.decoder = decoders.MLPGeneSpaceDecoder(input_length=input_length, hidden_size=self.hidden_size, num_layers=1, output_shape=output_shape, lr=self.learn_rate, device=device)
        self.genepool = genepool.GenePool(size=input_length, gsp=self.decoder, binary_mode=True)
        self.num_families = int(max(16, self.max_population // 28)) 
        print("families: ", self.num_families)
        self.num_children = 4

        self.environment = environments.Environment(layers=[
            layers.NPointCrossover(selection_function=selection.RankBasedSelection(amount_to_select=lambda: 2, factor = 20).select, families=self.num_families, children=self.num_children, n_points=8),
            layers.BinaryFlipMutation(selection_function=selection.RandomSelection(percent_to_select=1).select, flip_rate=0.1/self.scale),
        ], genepool=self.genepool, pbf_function=fitness_function)

    def solve(self, generations=300, backprop_mode=environments.BackpropMode.GRADIENT_DESCENT, backprop_every_n=10, epochs=1, selection_percent=.4, batch_size=32):
        self.environment.compile(start_population=self.max_population, max_individuals=self.max_population)
        self.environment.evolve(generations=generations, backprop_mode=backprop_mode, backprop_every_n=int(backprop_every_n * self.scale), epochs=epochs, selection_percent=selection_percent, batch_size=int(batch_size * self.scale))
    
    def plot_fitness(self):
        self.environment.plot()