# GeneSpace: A New Framework for AGE: Artificial General Evolution

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

In *genetic algorithms*, we are almost always directly evolving the phenotypes of our solutions. Even with algorithms that claim to seperate G-P mappings, it is usually pretty superficial. This leads to a *lot* of problems; I think these include:


- **Highly specific genetic algorithms**: Since every genetic algorithm works by directly evolving observable traits (phenotypes), it means there can be no *universal genetic code* like we have in biology (DNA). Each algorithm must then be specifically designed for each task, and there can be no sharing of genetic material between algorithms where the focus may be on different modalities.

- **Inefficiency**: Since we are only evolving the phenotypes of our solutions, we are greatly limited by the size of our phenotypes, and so we must, to our detriment, keep our populations low.

- **Bad Search Spaces**: In typical GAs, our search space is so wide that we must explore it very slowly. A single mutation usually results in a very small incremental change to our solution. In true language-based evolution, we see that a single mutation can actually have large changes to the phenotype. This also means that crossover, in many ways, is more like combining ideas rather than combining phenotypes. 

I have found one older paper that has a similar concept here (paper link)[https://link.springer.com/chapter/10.1007/978-3-319-10762-2_11] But I can't find any code, the focus seems a bit different, and the paper is more theoretical. This project is all code, and focuses on practical applications.

## How GeneSpace Solves These Problems

This project is brand new, but it is heavily based on Finch, a framework I have built over the past several years. While building it, and rebuilding it many times, I learned a lot about GAs and their limitations. My hope is that GeneSpace will build upon what I learned while building Finch, but also be a much more powerful, general, almost *linguistic* framework for artificial evolution. 

So here's GeneSpace:

### Universal "genespaces"
In DNA, everything is represented as a sequence of *A* *T* *C* *G*. In GeneSpace, we represent everything as a sequence of either binaries either *0* and *1*.

### GeneSpace Decoders

If our genotype of binary, how do we decode it into a phenotype? We use a decoder. Here it makes sense to use neural networks. (I almost called these gene regulatory networks).

We will call these decoders *GeneSpaceDecoders* (*/decoders.py*). They are neural networks that take a genotype and decode it into a phenotype. As the genetic algorithm is evolving these sequences of floats (inputs), we can use these decoders to generate a phenotype (output) of any size or shape. Since MLPs (Multilayer Perceptrons) are known as *universal function approximators*, we can use them to approximate any continuous function, including our decoders.

This means that over time, our GA will evolve genotypes that best decode into phenotypes, all while our *genespace* is learning to get better at producing the phenotypes from the genotypes. We can do this using backpropagation: mapping the worst genotypes to the best phenotypes.

This ends up creating a more flexible and powerful (co)evolutionary system. We only need one algorithm, one that works on inputs to the decoders, and we can decode it into any shape or size we want!

I have also been inspired by Stephen Wolfram's posts about "mining the mathematical universe" and as such, these algorithms focus on creating *genespace*(s) that just *happen* to work, and genes that just *happen* to exploit these neural networks appropriately. Without rhyme or reason really. I learned this could work when I accidentally discovered that genetic algorithms could *just so happen* to evolve prompts/images that trick the best image generation/recognition models into producing almost any desired output. This whole project is building on that discovery. 

## Current State

This project is still in its infancy. Below are some examples demonstrating the versatility of GeneSpace.

## Contributing

Please contribute! You can contact me on X/Twitter [@DanielJLosey](https://x.com/DanielJLosey).

## The Future

AGE: Artificial General Evolution. (Like AGI, but for evolution.)
Genetic algorithms that can evolve any arbitrary solution, with a universal -shareable- genetic code. A universal general genetic algorithm applicable to any problem. I hope this project will be a good step toward that goal. 

## License

This project is licensed under the MIT License. But I would appreciate it if you could give me a shout if you end up using this in a project!

# Examples

## General Evolution

GeneSpace introduces a `GeneralEvolution` class that can be used to solve various optimization problems using the same underlying algorithm. Here are two examples demonstrating its versatility:

### Image Evolution

In this example, we use GeneSpace to evolve an image to match a target image.

```python
from genespace import general_evolution
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch.nn.functional as F

def download_image(size):
    response = requests.get('https://static.wikia.nocookie.net/disney/images/6/64/Profile_-_Spider-Man.png/revision/latest?cb=20220320010954')
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(size)
    return np.array(img) / 255.0

X = 50  # Image size
image = download_image((X, X))
target_img_tensor = torch.from_numpy(image).float().to('cuda')

def image_fitness(phenotypes):
    target = target_img_tensor.unsqueeze(0).repeat(phenotypes.size(0), 1, 1, 1)
    mse = F.mse_loss(phenotypes, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1) 
    fitness = 1 / (mse + 1e-8)
    fitness_percentages = (fitness / (1 / 1e-8)) * 100
    return fitness_percentages.tolist()

image_ge = general_evolution.GeneralEvolution(image_fitness, 
                                              output_shape=(X, X, 3), 
                                              device="cuda", scale=2)

image_ge.solve(32000)

# Render the evolved image
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

render_evolved_image(image_ge.environment)
```
Note that the algorithm only takes up 2 lines!
```python
image_ge = general_evolution.GeneralEvolution(image_fitness, 
                                              output_shape=(X, X, 3), 
                                              device="cuda", scale=2)
image_ge.solve(32000)
```

### Traveling Salesman Problem (TSP)

In this example, we use the same `GeneralEvolution` class to solve the Traveling Salesman Problem.

```python
import numpy as np
import torch
from genespace import general_evolution
import matplotlib.pyplot as plt

# Define the cities
num_cities = 12
np.random.seed(42)
cities = np.random.rand(num_cities, 2)

# Compute the distance matrix
distance_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))

def city_fitness(phenotypes):
    batch_size = phenotypes.size(0)
    total_distances = []
    for i in range(batch_size):
        phenotype = phenotypes[i].cpu().numpy()
        route = np.argsort(phenotype)
        total_distance = 0.0
        for j in range(num_cities):
            from_city = route[j]
            to_city = route[(j + 1) % num_cities]
            total_distance += distance_matrix[from_city, to_city]
        total_distances.append(total_distance)
    fitness_values = 1 / (np.array(total_distances) + 1e-8)
    return fitness_values.tolist()

tsp_ge = general_evolution.GeneralEvolution(city_fitness, 
                                            output_shape=(num_cities,), 
                                            device="cuda", scale=2)

tsp_ge.solve(400)

# Plot the best route
def plot_route(tsp):
    best_genes = tsp.environment.individuals[0].genes
    best_genes_tensor = torch.tensor(best_genes, dtype=torch.float32).unsqueeze(0).to('cuda')

    with torch.no_grad():
      phenotype = tsp.environment.genepool.gsp.forward(best_genes_tensor)
    phenotype = phenotype.cpu().numpy().squeeze()
    route = np.argsort(phenotype)

    plt.figure(figsize=(8, 6))
    route_cities = cities[route]
    plt.plot(route_cities[:, 0], route_cities[:, 1], 'o-', label='Route')
    plt.plot([route_cities[-1, 0], route_cities[0, 0]], [route_cities[-1, 1], route_cities[0, 1]], 'o-')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], f'City {i}')
    plt.title('Best TSP Route Found')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

plot_route(tsp_ge)
```
Again, the algorithm only takes up 2 lines!
```python
tsp_ge = general_evolution.GeneralEvolution(city_fitness, 
                                            output_shape=(num_cities,), 
                                            device="cuda", scale=2)

tsp_ge.solve(400) # Less steps
```

With almost no modifications to the code, we can solve completely different problems. 

GeneSpaceDecoders.

Et. voila.
