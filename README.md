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

In *genetic algorithms*, we are almost always directly evolving the phenotypes of our solutions. This leads to a *lot* of problems; in my humble opinion, these include:


- **Highly specific genetic algorithms**: Since every genetic algorithm works by directly evolving observable traits (phenotypes), it means there can be no *universal genetic code* like we have in biology (DNA). Each algorithm must then be specifically designed for each task, and there can be no sharing of genetic material between algorithms where the focus may be on different modalities.

- **Inefficiency**: Since we are only evolving the phenotypes of our solutions, we are greatly limited by the size of our phenotypes, and so we must keep our populations low (I know this from experience). 

- **Complex Search Spaces**: In typical GAs, our search space is so wide that we must explore it very slowly. A single mutation usually results in a very small incremental change to our solution. In true language-based evolution, we see that a single mutation can actually have large changes to the phenotype. This also means that crossover, in many ways, is more like combining ideas rather than combining phenotypes. 

I have found one older paper that has a similar concept here (paper link)[https://link.springer.com/chapter/10.1007/978-3-319-10762-2_11] But I can't find any code, the focus seems a bit different, and the paper is more theoretical. This project is all code, and focuses on practical applications.

## How GeneSpace Solves These Problems

This project is brand new, but it is heavily based on Finch, a framework I have built over the past several years. While building it, I learned a lot about GAs and their limitations. My hope is that GeneSpace will build upon what I learned while building Finch, but also be a much more powerful, general, *linguistic* framework for artificial evolution. 

So here's GeneSpace:

### Universal Genespace
In DNA, everything is represented as a sequence of *A* *T* *C* *G*. In GeneSpace, we represent everything as a sequence of floats between *0* and *1* (or optionally binary). It is good to mimic the ideas behind biology, but not necessarily the implementation itself. We learned this lesson especially with the success of Deep Learning.

### GeneSpace Decoders

If we use a genotype of binary floats, how do we decode it into a phenotype? We use a decoder. What better decoder exists than a neural network? (I almost called these gene regulatory networks, which are a thing in biology).

We will call these decoders *GeneSpaceDecoders* (*/decoders.py*). They are neural networks that take a genotype and decode it into a phenotype. As the genetic algorithm is evolving these sequences of floats, we can use these decoders to generate a phenotype of any size or shape. Since MLPs (Multilayer Perceptrons) are known as *universal function approximators*, we can use them to approximate any function, including our decoders.

This means that over time, our GA will evolve genotypes that best decode into phenotypes, all while our *genespace* is learning to get better at producing the phenotypes from the genotypes. We can do this using two different techniques implemented in our @decoders.py:

1. Backpropagation: The `backprop_network` method in our `GeneSpaceDecoderBase` class allows us to train the decoder network using the top-performing individuals as targets for the lower-performing ones. This helps the decoder learn to map genotypes to successful phenotypes. It also encourages genetic diversity.

2. Random Gradient Application (my favorite): The `apply_random_gradient` method generates random gradients, applies them to the network, and keeps the one that produces the best fitness improvement. This introduces controlled randomness into the learning process, potentially helping to escape local optima.

These techniques allow our GeneSpace to adaptively evolve the mapping between genotypes and phenotypes, creating a more flexible and powerful (co)evolutionary system. Going forward, I want to move away from backpropagation as much as possible.

I have also been inspired by Stephen Wolfram's posts about "mining the mathematical universe" and as such, these algorithms focus on creating *genespace*(s) that just *happen* to work, and genes that just *happen* to exploit these neural networks appropriately. Without rhyme or reason really. I learned this could work when I discovered that genetic algorithms could *just so happen* to evolve prompts/images that trick the best image generation/recognition models into producing almost any desired output. This whole project is building on that discovery. 

## Current State

This project is still in its infancy. I will post examples, colab notebooks, etc., in the coming weeks. 

## Contributing

Please contribute! You can contact me on X/Twitter [@DanielJLosey](https://x.com/DanielJLosey).


## The Future

Genetic algorithms that can evolve any arbitrary solution, with a universal -shareable- genetic code. A universal general genetic algorithm applicable to any problem. I hope this project will be a good step toward that goal. 

## License

This project is licensed under the MIT License. See the LICENSE file for more details. But I would appreciate it if you could give me a shout if you end up using this in a project!
