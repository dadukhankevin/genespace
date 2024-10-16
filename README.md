# GeneSpace

GeneSpace is a genetic algorithm framework that aims to take the best concepts from *actual evolution* and bring them to the field of evolutionary computation.

## Installation
```bash
git clone https://github.com/dadukhankevin/genespace
```

## About

## Background: Phenotypes vs Genotypes
I'm no expert in biology, but as you likely remember from 7th grade science, our existence can be explained in two ways:

- our genotype (our DNA)
- our phenotype (our observable traits)


Interestingly, in biology, multiple genotypes can lead to the same phenotype. This leads to interesting concepts
like convergent evolution, where two different species evolve similar traits in response to similar environmental pressures -- even with completely different underlying DNA (genotypes).

On the other hand, sometimes the same genotype (DNA) can lead to different phenotypes (observable traits). This hints that there are some evolvable traits that are not encoded in our DNA (genotype), but are instead encoded in some other (possibly non-genetic) format. In biology, this is called*epigenetics*, which involves changes in phenotype without altering the DNA sequence itself. 

## The Problem with (current) Evolutionary Algorithms

The difference between genotypes and phenotypes is fundamental to how biology works, and how evolution works in biology. Imagine if DNA and our observable traits had a 1-1 relationship. It would mean every cell in our body would need DNA telling it precisely where to be specifically, what atoms it requires, etc. Instead, we see that this is not the case. DNA is a highly compressed format for information, and our phenotypes are a result of this compressed information being *expressed/interpreted* in a specific way.

In *genetic algorithms*, we are almost always directly evolving the phenotypes of our solutions. This leads to a *lot* of problems; I think these include:


- **Highly specific genetic algorithms**: Since every genetic algorithm works by directly evolving observable traits (phenotypes), it means there can be no *universal genetic code* like we have in biology (DNA). Each algorithm must then be specifically designed for each task, and there can be no sharing of genetic material between algorithms where the focus may be on different modalities.

- **Inefficiency**: Since we are only evolving the phenotypes of our solutions, we are greatly limited by the size of our phenotypes, and so we must keep our populations low (I know this from experience). 

- **Bad Search Spaces**: In typical GAs, our search space is so wide that we must explore it very slowly. A single mutation usually results in a very small incremental change to our solution. In true language-based evolution, we see that a single mutation can actually have large changes to the phenotype. This also means that crossover, in many ways, is more like combining ideas rather than combining phenotypes. 

I have found one older paper that has a similar concept here (paper link)[https://link.springer.com/chapter/10.1007/978-3-319-10762-2_11] But I can't find any code, the focus seems a bit different, and the paper is more theoretical. This project is all code, and focuses on practical applications.

## How GeneSpace Solves These Problems

This project is brand new, but it is heavily based on Finch, a framework I have built over the past several years. While building it, and rebuilding it many times, I learned a lot about GAs and their limitations. My hope is that GeneSpace will build upon what I learned while building Finch, but also be a much more powerful, general, almost *linguistic* framework for artificial evolution. 

So here's GeneSpace:

### Universal Genespace
In DNA, everything is represented as a sequence of *A* *T* *C* *G*. In GeneSpace, we represent everything as a sequence of floats between *0* and *1* (or optionally binary). It is good to mimic the ideas behind biology, but not necessarily the implementation itself. We learned this lesson especially with the success of Deep Learning.

### GeneSpace Decoders

If we use a genotype of binary floats, how do we decode it into a phenotype? We use a decoder. Here it makes sense to use neural networks. (I almost called these gene regulatory networks, which are a thing in biology).

We will call these decoders *GeneSpaceDecoders* (*/decoders.py*). They are neural networks that take a genotype and decode it into a phenotype. As the genetic algorithm is evolving these sequences of floats (inputs), we can use these decoders to generate a phenotype (output) of any size or shape. Since MLPs (Multilayer Perceptrons) are known as *universal function approximators*, we can use them to approximate any continuous function, including our decoders.

This means that over time, our GA will evolve genotypes that best decode into phenotypes, all while our *genespace* is learning to get better at producing the phenotypes from the genotypes. We can do this using two different techniques implemented in our @decoders.py:

1. Backpropagation: The `backprop_network` method in our `GeneSpaceDecoderBase` class allows us to train the decoder network using the top-performing individuals as targets for the lower-performing ones (or other similar techniques). This helps the decoder learn to map genotypes to successful phenotypes. It also encourages genetic diversity.

2. Random Gradient Application: The `apply_random_gradient` method generates random gradients, applies them to the network, and keeps the one that produces the best fitness improvement. This introduces controlled randomness into the learning process, potentially helping to escape local optima. Its a bit simplistic for now, but I think we can do better.

These techniques allow our GeneSpace to adaptively evolve the mapping between genotypes and phenotypes, creating a more flexible and powerful (co)evolutionary system. Going forward, I want to move away from backpropagation as much as possible.

I have also been inspired by Stephen Wolfram's posts about "mining the mathematical universe" and as such, these algorithms focus on creating *genespace*(s) that just *happen* to work, and genes that just *happen* to exploit these neural networks appropriately. Without rhyme or reason really. I learned this could work when I accidentally discovered that genetic algorithms could *just so happen* to evolve prompts/images that trick the best image generation/recognition models into producing almost any desired output. This whole project is building on that discovery. 

## Current State

This project is still in its infancy. I will post examples, colab notebooks, etc., in the coming weeks. 

## Contributing

Please contribute! You can contact me on X/Twitter [@DanielJLosey](https://x.com/DanielJLosey).


## The Future

Genetic algorithms that can evolve any arbitrary solution, with a universal -shareable- genetic code. A universal general genetic algorithm applicable to any problem. I hope this project will be a good step toward that goal. 

## License

This project is licensed under the MIT License. See the LICENSE file for more details. But I would appreciate it if you could give me a shout if you end up using this in a project!

# Examples

## Image Evolution Demo

Credit goes to PyGad for their image reconstruction demo as inspiration! Here we will take a similar concept, but use our neural networks to decode genotypes into phenotypes.

***Note:*** Since the fitness function only has one perfect maximum, this is a bad problem to truly use genetic algorithms for, most GAs work better with large solution spaces. However since this example is visual (images) its nice and fun to test with.


### Table of Contents
1. [Setup](#setup)
2. [Downloading a Target Image](#downloading-a-target-image)
3. [Defining the Fitness Function](#defining-the-fitness-function)
4. [Creating the Environment](#creating-the-environment)
5. [Evolving the Image](#evolving-the-image)
6. [Visualizing Results](#visualizing-results)

### Setup

First, clone the GeneSpace repository:

```bash
git clone https://github.com/dadukhankevin/genespace
```

Install the required dependencies:

```bash
pip install pillow requests numpy torch matplotlib
```

### Downloading a Target Image

We'll start by downloading a target image that our genetic algorithm will try to evolve towards:

```python
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch

def download_image(size):
    response = requests.get('https://static.wikia.nocookie.net/disney/images/6/64/Profile_-_Spider-Man.png/revision/latest?cb=20220320010954')
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(size)
    return np.array(img) / 255.0

X = 50
image = download_image((X, X))
target_img_tensor = torch.from_numpy(image).float().to('cuda')

# Display the image
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis('off')
plt.title('Target Image')
plt.show()
```

### Defining the Fitness Function

The fitness function evaluates how close each evolved image is to the target image. This is the *only* metric our genetic algorithm will use.

```python
import torch.nn.functional as F

def batch_pheno_fitness(phenotypes):
    target = target_img_tensor.unsqueeze(0).repeat(phenotypes.size(0), 1, 1, 1)
    
    if phenotypes.shape != target.shape:
        phenotypes = phenotypes.permute(0, 3, 1, 2)
    
    mse = F.mse_loss(phenotypes, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    
    fitness = 1 / (mse + 1e-8)
    max_fitness = 1 / 1e-8
    fitness_percentages = (fitness / max_fitness) * 100
    
    return fitness_percentages.tolist()
```

### Creating the Environment

Set up the GeneSpace environment:

```python
from genespace import layers, genepool, decoders, selection, environments
import torch.nn as nn

cross_selection = selection.RankBasedSelection(amount_to_select=lambda: 2, factor=40)
insertion_selection = selection.RankBasedSelection(amount_to_select=200, factor=-1)

input_length = 1000
output_shape = (X, X, 3)

training_mode = decoders.TrainingMode.TOP_AND_BOTTOM_PERCENT
gene_space_network = decoders.MLPGeneSpaceDecoder(
    input_length=input_length,
    hidden_size=20000,
    num_layers=1,
    output_shape=output_shape,
    device='cuda',
    lr=.0001,
    activation=nn.LeakyReLU,
    training_mode=training_mode
)

mode = environments.BackpropMode.GRADIENT_DESCENT
gene_pool = genepool.GenePool(size=input_length, gsp=gene_space_network, binary_mode=True)

environment = environments.Environment(
    layers=[
        layers.NPointCrossover(selection_function=cross_selection.select, families=16, children=4, n_points=8),
        layers.BinaryFlipMutation(selection_function=insertion_selection.select, flip_rate=.05)
    ],
    genepool=gene_pool,
    pbf_function=batch_pheno_fitness
)

environment.compile(start_population=10, max_individuals=200)
```

### Evolving the Image

Run the evolution process:

```python
environment.evolve(generations=5000, epochs=1, backprop_every_n=10, selection_percent=.2, backprop_mode=mode, batch_size=128)

environment.plot()
```

### Visualizing Results

Render and display the evolved image. Here we take the genes of our best individual, pass them through our decoder, then display the result!

```python
def render_evolved_image(environment, index=0):
    genes = environment.individuals[index].genes
    genes_tensor = torch.tensor(genes, dtype=torch.float32).to(environment.genepool.gsp.device)
    genes_tensor = genes_tensor.unsqueeze(0)
    
    with torch.no_grad():
        phenotype = environment.genepool.gsp.forward(genes_tensor)
    
    image_array = phenotype.squeeze().cpu().numpy()
    image_array = np.clip(image_array, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array)
    plt.axis('off')
    plt.title(f'Evolved Image (Individual {index})')
    plt.show()

# Display target and evolved images
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis('off')
plt.title('Target Image')
plt.show()

render_evolved_image(environment)

print(environment.individuals[0].genes)
```

Feel free to experiment with different parameters, fitness functions, or target images to see how they affect the evolution process! I'm sure there are better ways to set this up. 


## GeneSpace Backpack Problem Tutorial

This tutorial demonstrates how to use the GeneSpace framework to solve the classic backpack problem. We'll use a genetic algorithm to optimize item selection for a backpack, considering weight limits, item values, and category constraints.

### Table of Contents
1. [Setup](#setup)
2. [Generating Random Items](#generating-random-items)
3. [Defining the Fitness Function](#defining-the-fitness-function)
4. [Configuring the Genetic Algorithm](#configuring-the-genetic-algorithm)
5. [Creating the Environment](#creating-the-environment)
6. [Running the Evolution](#running-the-evolution)
7. [Analyzing Results](#analyzing-results)

### Setup

First, ensure you have the GeneSpace framework installed. If not, you can clone it from the repository:

```bash
git clone https://github.com/dadukhankevin/genespace
```

Import the necessary modules:

```python
from genespace import layers, genepool, decoders, selection, environments
import numpy as np
import torch
import random
import string
```

### Generating Random Items

We'll start by creating a function to generate random items for our backpack problem:

```python
def generate_random_items(num_items, min_weight=0.1, max_weight=10.0, min_value=1, max_value=10, num_categories=8):
    categories = list(string.ascii_uppercase[:num_categories])
    
    items = []
    for _ in range(num_items):
        weight = round(random.uniform(min_weight, max_weight), 2)
        value = random.randint(min_value, max_value)
        category = random.choice(categories)
        items.append((weight, value, category))
    
    return items

num_items = 50
items = generate_random_items(num_items)
```

### Defining the Fitness Function

The fitness function evaluates how good a solution is, considering the total value, weight limit, and category constraints:

```python
max_weight = 9  # Maximum weight the backpack can hold

def backpack_fitness(phenotypes):
    fitnesses = []
    for phenotype in phenotypes:
        total_weight = 0
        total_value = 0
        selected_categories = set()
        penalty = 1

        for i, (weight, value, category) in enumerate(items):
            if phenotype[i] > 0.5:  # If the item is selected
                if category in selected_categories:
                    penalty += value * 2  # Penalty for duplicate category
                else:
                    selected_categories.add(category)
                    total_weight += weight
                    total_value += value

        # Apply penalty for exceeding weight limit
        if total_weight > max_weight:
            overweight = total_weight - max_weight
            penalty += overweight * 2

        fitness = total_value / penalty
        fitnesses.append(fitness)

    return fitnesses
```

### Configuring the Genetic Algorithm

Set up the parameters for the genetic algorithm:

```python
GENE_LENGTH = 250
HIDDEN_SIZE = 256 * 10
OUTPUT_SHAPE = (len(items),)
LEARNING_RATE = 0.00001
DEVICE = 'cpu'
LAYERS = 1

CROSSOVER = selection.RankBasedSelection(amount_to_select=2, factor=20)
MUTATION = selection.RandomSelection(percent_to_select=1)
MUTATION_MAGNITUDE = 0.1
N_FAMILIES = 16
CHILDREN = 4
N_POINTS = 8

GENE_SPACE = decoders.MLPGeneSpaceDecoder(
    input_length=GENE_LENGTH,
    hidden_size=HIDDEN_SIZE,
    num_layers=LAYERS,
    output_shape=OUTPUT_SHAPE,
    device=DEVICE,
    lr=LEARNING_RATE,
)

GENE_POOL = genepool.GenePool(size=GENE_LENGTH, gsp=GENE_SPACE, binary_mode=True)
```

### Creating the Environment

Set up the GeneSpace environment:

```python
environment = environments.Environment(
    layers=[
        layers.NPointCrossover(selection_function=CROSSOVER.select, families=N_FAMILIES, children=CHILDREN, n_points=N_POINTS),
        layers.BinaryFlipMutation(selection_function=MUTATION.select, flip_rate=MUTATION_MAGNITUDE)
    ],
    genepool=GENE_POOL,
    pbf_function=backpack_fitness
)

environment.compile(start_population=200, max_individuals=200)
```

### Running the Evolution

Execute the genetic algorithm:

```python
_ = environment.evolve(generations=300, backprop_every_n=10, selection_percent=.4)

environment.plot()
```

### Analyzing Results

After evolution, we can analyze the best solution:

```python
best_individual = environment.individuals[0]
best_phenotype = GENE_SPACE.forward(torch.tensor(best_individual.genes, dtype=torch.float32).unsqueeze(0)).squeeze().detach().numpy()

print("Best solution:")
total_weight = 0
total_value = 0
selected_categories = set()
for i, (weight, value, category) in enumerate(items):
    if best_phenotype[i] > 0.5:
        print(f"Item {i+1}: Weight {weight}, Value {value}, Category {category}")
        total_weight += weight
        total_value += value
        selected_categories.add(category)

print(f"\nTotal weight: {total_weight}")
print(f"Total value: {total_value}")
print(f"Number of unique categories: {len(selected_categories)}")

# Print the genes of the best individual
print("Genes of the best individual:")
print(environment.individuals[0].genes)
```

This tutorial demonstrates how to use GeneSpace to solve the backpack problem, a classic optimization challenge. By running this code, you'll see how the genetic algorithm evolves a population of solutions to find an optimal or near-optimal selection of items for the backpack, considering weight limits, item values, and category constraints. 

Key point: Overall the algorithm for both the Image Evolution and Backpack Problem are very similar. Both use the same two layers, and both use a GeneSpaceDecoder to decode genotypes into phenotypes. 