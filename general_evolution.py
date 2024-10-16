# Experimental 
from genespace import layers, genepool, decoders, selection, environments
import enum

class GeneralEvolution:
    """
    A class for general evolutionary computation using the GeneSpace framework.

    This class sets up and manages the evolutionary process, including the genetic algorithm
    components and the neural network decoder.

    Attributes:
        scale (float): A scaling factor for various parameters.
        hidden_size (int): The size of hidden layers in the neural network decoder.
        learn_rate (float): The learning rate for the neural network decoder.
        max_population (int): The maximum population size for the genetic algorithm.
        decoder (decoders.MLPGeneSpaceDecoder): The neural network decoder for genotype-to-phenotype mapping.
        genepool (genepool.GenePool): The gene pool managing genetic information.
        num_families (int): The number of families for crossover operations.
        num_children (int): The number of children produced per family.
        environment (environments.Environment): The evolutionary environment managing the process.
    """

    def __init__(self, fitness_function, output_shape: tuple[int, ...], scale: float=1, input_length: int = 250, device: str = "cpu"):
        """
        Initialize the GeneralEvolution instance.

        Args:
            fitness_function (callable): The fitness function to evaluate individuals.
            output_shape (tuple[int, ...]): The shape of the output (phenotype).
            scale (float, optional): Scaling factor for various parameters. Defaults to 1.
            input_length (int, optional): Length of the input genotype. Defaults to 250.
            device (str, optional): The device to run computations on ('cpu' or 'cuda'). Defaults to "cpu".
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
        """
        Run the evolutionary process to solve the problem.

        Args:
            generations (int, optional): Number of generations to evolve. Defaults to 300.
            backprop_mode (environments.BackpropMode, optional): Mode for backpropagation. Defaults to GRADIENT_DESCENT.
            backprop_every_n (int, optional): Frequency of backpropagation. Defaults to 10.
            epochs (int, optional): Number of epochs for each backpropagation. Defaults to 1.
            selection_percent (float, optional): Percentage of population to select for reproduction. Defaults to 0.4.
            batch_size (int, optional): Batch size for backpropagation. Defaults to 32.
        """
        self.environment.compile(start_population=self.max_population, max_individuals=self.max_population)
        self.environment.evolve(generations=generations, backprop_mode=backprop_mode, backprop_every_n=int(backprop_every_n * self.scale), epochs=epochs, selection_percent=selection_percent, batch_size=int(batch_size * self.scale))
    
    def plot_fitness(self):
        """
        Plot the fitness progression over generations.
        """
        self.environment.plot()