# GeneSpace

GeneSpace is a new experimental framework that unifies Genetic Algorithms (GAs) and Neural Networks by leveraging Neural Networks as Gene Regulatory Networks (GRNs).

 This integration allows for the evolution of genotypes into complex phenotypes, enabling sophisticated mappings and optimizations. GeneSpace facilitates the co-evolution of genetic representations and their corresponding neural network decoders, enhancing the adaptability and efficiency of evolutionary processes.


#### Key point: Until now GAs have opperated by directly evolving phenotypes (e.g. images, audio, etc.). This framework allows for the evolution of genotypes (i.e. the genetic code itself), which can be decoded into phenotypes by a neural network. This is much more similar to biology!

I'm super excited about this. We are close to artificial general intelligence, now that every 'individual' can be creating useing the same genetic code (aka genespace). 


## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
  - [Classes](#classes)
    - [GeneRegulatoryNetwork](#gene_regulatory_network)
    - [Individual](#individual)
    - [GenePool](#genepool)
    - [Layer](#layer)
      - [NPointCrossover](#npointcrossover)
      - [UniformMutation](#uniformmutation)
    - [Selection Strategies](#selection-strategies)
      - [RandomSelection](#randomselection)
      - [TournamentSelection](#tournamentselection)
      - [RankBasedSelection](#rankbasedselection)
    - [Environment](#environment)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Example Projects](#example-projects)
  - [Example](#evolving-image-like-phenotypes) Better examples are coming!
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Unified Framework**: Allows for advanced genotype-to-phenotype mappings. Introduces many -> many relationships where multiple genotypes can map to the same phenotype and vice versa. 
- **Modular Design**: Well-structured classes and layers help with easy extension and experimentation. It is based on my previous project Finch, but with many improvements and much more experimental.
- **Flexible Selection and Genetic Operators**: Supports various selection strategies and genetic operations.
- **Simpler**: Now that all genetic algorithms can be represnted using the same genespace (sequences of floats between 0 and 1), we can use the same genetic operators for all GAs.
- **Backpropagation Integration**: Incorporates gradient-based optimization to train the GRN alongside the evolutionary process, fostering co-evolution of genes and their decoders. Individuals increasingly produce better and better phenotypes while the GRN becomes more and more adept at decoding the same individuals.
- **Batch Processing**: Efficiently handles large populations through batch processing, optimizing computational resources. NVIDIA is going to love this (:

## Architecture

GeneSpace is composed of several interrelated classes that manage different aspects of the evolutionary process. Below is an overview of the core components.

### Classes

#### GeneRegulatoryNetwork

**File**: `grn.py`

The `GeneRegulatoryNetwork` class extends PyTorch's `nn.Module` and serves as the neural network that maps genotypes to phenotypes. It encapsulates the neural network architecture, training mechanisms, and backpropagation logic.

**Key Features**:

- **Customizable Architecture**: Configurable input length, hidden sizes, number of layers, and output shapes. (Basic MLP, I suspect this is all we need.)
- **Device Management**: Supports CPU, GPU.
- **Backpropagation Modes**: Implements multiple training strategies (`divide_and_conquer`, `weighted`, `direct_prediction`) to train the GRN based on the population's fitness. I'm not sure whcih is best yet, but I suspect divide_and_conquer better overall, and ensures genetic diversity.
- **Optimizer and Loss Function**: Incorporates optimizers (e.g., Adam) and loss functions (e.g., Mean Squared Error) for training.

**Methods**:

- `forward(x)`: Processes input genotypes through the neural network to produce phenotypes.
- `backprop_network(individuals, target_image, mode)`: Trains the GRN based on the specified mode using the current population of individuals.

#### Individual

**File**: `individual.py`

Represents an individual in the population, encapsulating its genotype, fitness score, and modification status.

**Attributes**:

- `genes`: NumPy array representing the genotype.
- `fitness`: Float value indicating the individual's fitness.
- `modified`: Boolean flag indicating if the individual has been modified since the last fitness evaluation.

**Methods**:

- **None**: The class primarily serves as a data container without callable methods to maintain batch processing capabilities.

#### GenePool

**File**: `genepool.py`

Manages the creation and handling of individuals within the population.

**Key Features**:

- **Gene Initialization**: Supports both binary and real-valued gene representations.
- **Individual Creation**: Generates new individuals with randomized genes.

**Methods**:

- `create_genes()`: Generates a new gene sequence.
- `generate_one_gene()`: Generates a single gene value.
- `create_individual()`: Instantiates a new `Individual` with generated genes.

#### Layer

**File**: `layers.py`

Abstract base class for genetic operators that manipulate the population through crossover and mutation.

**Subclasses**:

##### NPointCrossover

**File**: `layers.py`

Performs N-point crossover between selected parents to produce offspring.

**Parameters**:

- `selection_function`: Callable to select parent individuals.
- `families`: Number of parent pairs to process.
- `children`: Number of offspring per parent pair.
- `n_points`: Number of crossover points.
- `device`: Computation device (`cpu` or `gpu`).

**Methods**:

- `execute()`: Executes the crossover operation and returns new offspring individuals.

##### UniformMutation

**File**: `layers.py`

Applies uniform mutation to selected individuals by adding random noise.

**Parameters**:

- `selection_function`: Callable to select individuals for mutation.
- `device`: Computation device (`cpu` or `gpu`).
- `magnitude`: Magnitude of the mutation noise.

**Methods**:

- `execute()`: Executes the mutation operation and returns mutated individuals.

#### Selection Strategies

**File**: `selection.py`

Implements various selection strategies to choose individuals for reproduction and mutation.

##### RandomSelection

**File**: `selection.py`

Selects individuals randomly from the population.

**Parameters**:

- `percent_to_select`: Callable returning the percentage of individuals to select.
- `amount_to_select`: Callable returning the number of individuals to select.

**Methods**:

- `select(individuals)`: Returns a list of randomly selected individuals.

##### TournamentSelection

**File**: `selection.py`

Selects individuals based on tournament selection, favoring higher fitness.

**Parameters**:

- `percent_to_select`: Callable returning the percentage of individuals to select.
- `amount_to_select`: Callable returning the number of individuals to select.

**Methods**:

- `select(individuals)`: Returns a list of selected individuals through tournament selection.

##### RankBasedSelection

**File**: `selection.py`

Selects individuals based on their rank, applying a selection pressure factor.

**Parameters**:

- `factor`: Selection pressure factor.
- `percent_to_select`: Callable returning the percentage of individuals to select.
- `amount_to_select`: Callable returning the number of individuals to select.

**Methods**:

- `select(individuals)`: Returns a list of selected individuals based on rank-based probabilities.

#### Environment

**File**: `environments.py`

Manages the evolutionary process, integrating genetic operators, fitness evaluations, and training of the GRN.

**Key Features**:

- **Layer Integration**: Incorporates genetic operators (layers) to manipulate the population.
- **Fitness Evaluation**: Calculates fitness scores using batch processing for efficiency.
- **GRN Training**: Invokes the GRN's backpropagation method to train the network based on the current population.
- **Evolution Loop**: Handles the main evolutionary loop over specified generations.
- **Visualization**: Provides plotting capabilities to visualize fitness and population trends.

**Methods**:

- `compile(start_population, max_individuals, batch_size, individuals, early_stop)`: Initializes the environment with population parameters.
- `batch_fitness()`: Evaluates the fitness of modified individuals in batches.
- `sort_individuals()`: Sorts individuals based on fitness.
- `evolve(generations, backprop_mode)`: Executes the evolutionary process over a number of generations.
- `plot()`: Visualizes the fitness and population history.

## Getting Started

Follow these instructions to set up and run the GeneSpace project on your local machine.

### Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **pip** (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/genespace.git
   cd genespace
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If `requirements.txt` is not provided (its not yet lol), install the necessary packages manually:

   ```bash
   pip install torch numpy pillow matplotlib
   ```
etc..
### Usage

GeneSpace can be utilized to evolve populations for various tasks. Below is a simple example. Colab notebooks will be added soon.

#### Running the Image Evolution Example

1. **Prepare the Project Structure**

   Ensure that all classes and utility functions are correctly placed in their respective files:

   ```
   genespace/
   ├── grn.py
   ├── individual.py
   ├── genepool.py
   ├── layers.py
   ├── selection.py
   ├── environments.py
   ├── fitness.py
   ├── utils.py
   ├── example.py
   └── README.md (you are here haha)
   ```

2. **Execute the Example Script**

   ```bash
   python example.py
   ```

   **What Happens**:

   - Creates a population of individuals with random genes.
   - Evaluates the fitness of each individual.
   - Creates a gene regulatory network and trains it on the population.
   - Evolves the population over 100 generations.
   - Visualizes the fitness and population history.
   - Maximizes the sum of the phenotypes.


## Project Structure

```
genespace/
├── grn.py                # GeneRegulatoryNetwork class
├── individual.py         # Individual class
├── genepool.py           # GenePool class
├── layers.py             # Genetic operators (crossover, mutation)
├── selection.py          # Selection strategies
├── environments.py       # Environment class managing evolution
├── fitness.py            # Fitness functions
├── example.py            # Example script for evolving larger sum arrays. Super simple rn...
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE). But if you use it for something big, let me know so I can buy you a coffee! Also some credit is always nice (:

## Acknowledgements

- **PyTorch**: For providing a robust neural network framework.
- **NumPy**: For efficient numerical computations.
- **Pillow**: For image processing capabilities.
- **Matplotlib**: For visualization tools.
- **Requests**: For handling HTTP requests to download images.
- **Finch**: My previous project that this is heavily based on. Check it out!
- **X/Twitter**: For everyone who has followed me on this journey. It means a lot. Especially those who have offered encouragement, advice, or criticism.
- **Everyone**: For all the hard work that has gone into the fields of evolutionary algorithms, genetic programming, and neural networks. It's an blast to build in this field.

Happy evolving!