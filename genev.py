import time
import math
import sys
import copy
import platform
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import gridspec

_DEFAULT_FITNESS = sys.maxsize
_IDS = 0


class Individual:

    def __init__(self, structure, dna_skeleton):

        global _IDS
        self.id = _IDS
        _IDS += 1

        self.generation = 0
        self.structure = structure
        self.skeleton = dna_skeleton
        self.fitness_function = None

        self.dna = self.pickDNA(dna_skeleton)
        self.obj = structure(**self.dna)

        self.fitness = _DEFAULT_FITNESS
        self.unfit = False

    def attachFitnessFunction(self, fitness_function):
        self.fitness_function = fitness_function
        self.calcFitness = lambda: fitness_function(self.obj)

    def pickDNA(self, opts):
        # cast and call lambda function
        dna = {key: opts[key][0](opts[key][1]()) for key in opts.keys()}
        return dna

    def calcFitness(self):
        print("Error: Non-implemented function. Please call self.attachFitnessFunction from the Evolution component")
        print("Individual id #{}".format(self.id))
        return

    def updateFitness(self):
        self.fitness = self.calcFitness()
        return self.fitness

    def mate(self, individual):

        # Copy and shuffle keys for a random DNA fusion
        dna1, dna2 = dict(self.dna), dict(individual.dna)
        keys, klength = list(self.dna.keys()), len(dna1)
        np.random.shuffle(keys)

        # Switch key values
        for i, key in enumerate(keys):

            if i > klength / 2:
                break

            dna1[key] = individual.dna[key]
            dna2[key] = self.dna[key]

        child1 = Individual(self.structure, self.skeleton)
        child1.attachFitnessFunction(self.fitness_function)
        child1.dna = dna1

        child2 = Individual(individual.structure, individual.skeleton)
        child2.attachFitnessFunction(individual.fitness_function)
        child2.dna = dna2

        return child1, child2

    def mutate(self, proba):

        p_range = (1 - proba / 2, 1 + proba / 2)

        # Change randomly a little of every DNA parts
        for key in self.dna.keys():
            if self.skeleton[key][0] is not str:
                self.dna[key] *= np.random.uniform(p_range[0], p_range[1])
                self.dna[key] = self.skeleton[key][0](self.dna[key])  # cast to the right type

                # Update the object itself
                setattr(self, key, self.dna[key])

    def __str__(self):
        return "[#{} / gen {}]\tscore is {}".format(self.id, self.generation, self.fitness)


class Evolution:

    def __init__(self, n_population, structure, dna_skeleton, epuration_factor=0.5, mutation_probability=0.8, mutation_range=0.4):

        global _IDS
        _IDS = 0

        self.n_population = n_population
        self.model_properties = {}

        self.elite = None
        self.population = []
        self.generation = 0
        self.fitnesses = {
            "total": _DEFAULT_FITNESS,
            "mean": _DEFAULT_FITNESS,
            "min": _DEFAULT_FITNESS,
            "max": 0
        }

        self.skeleton_stats = {key: [] for key in dna_skeleton.keys()}

        self.epuration_factor = epuration_factor
        self.mutation_probability = mutation_probability
        self.mutation_range = mutation_range

        self.count = 0  # used to monitor the avancement in evaluation

    def model(self, structure, dna_skeleton, fitness_function):
        self.model_properties = {
            "structure": structure,
            "dna_skeleton": dna_skeleton,
            "fitness_function": fitness_function
        }

    def create(self):

        i = len(self.population)
        while i < self.n_population:
            elem = Individual(
                self.model_properties["structure"], self.model_properties["dna_skeleton"])
            elem.attachFitnessFunction(
                self.model_properties["fitness_function"])
            elem.generation = self.generation
            self.population.append(elem)
            i += 1

    def sort(self):
        self.population.sort(key=lambda elem: elem.fitness)

    def evaluate(self, evaluate_elite=False, display=False):

        # The most performance-horrible function in this script:
        # Computing the fitness of each individual
        # It is called synchronously on Windows and asynchronously on Linux
        self.count = 0

        # # if we work on Linux, let's put in place an asynchronous call
        # if platform.system() != "Windows":
        #     pool = mp.Pool()
        #     print("update_individual", update_individual)
        #     print("display_progress", display_progress)
        #     print("display_error", display_error)
        #     print("self", self, "\n")
        #     for elem in self.population:
        #         pool.apply_async(update_individual,
        #                          args=(1,), callback=display_progress,
        #                          error_callback=display_error)
        #     pool.close()
        #     pool.join()

        display = lambda x, end="": print("\revaluation: {2:.2%}\t({0:} over {1:})".format(self.count, self.n_population, self.count/self.n_population), end=end)

        # # Fallback on synchronous call
        # else:
        for index, elem in enumerate(self.population):
            
            display(self.count)
            sys.stdout.flush()
            self.count += 1

            # Do not waste time to recompute the elite, which was not mutated
            if evaluate_elite is False and index == 0:
                continue

            elem.updateFitness()

        display(self.count, "\n")

        # Update the statistics
        fitnesses = [elem.fitness for elem in self.population]
        self.fitnesses["min"] = np.min(fitnesses)
        self.fitnesses["max"] = np.max(fitnesses)
        self.fitnesses["total"] = np.sum(fitnesses)
        self.fitnesses["total"] = self.fitnesses["total"] / self.n_population

        # Register skeleton stats to visualize "features"
        for elem in self.population:
            for key in self.skeleton_stats.keys():
                self.skeleton_stats[key].append([elem.fitness, elem.dna[key]])

        # Thanks to the evaluation, sort the array from the best to the worst individual
        self.sort();
        self.elite = self.population[0]

        # Display the whole population
        if display is True:
            for p in self.population:
                print(p)

    def select(self):
        '''
            -- Selection of the two parents --
            Actually we take the two best parents, but if they are equals we take somebody else in population to mate with the best one
            The fact to mate with some random individual is good for randomness.
            Can be improved with more randoms ?
        '''

        if (self.population[0].fitness == self.population[1].fitness):
            return self.population[0], self.population[int(np.random.uniform(0, self.n_population))]

        return self.population[0:2]

    def generate(self, parents):

        gen = self.population
        self.generation += 1
        elite = parents[0]

        # Make a good generation with two selected individuals
        children = parents[0].mate(parents[1])

        for child in children:
            child.generation = self.generation
            child.updateFitness()

            # Avoid a too similar couple parents/children
            if child.fitness == parents[0].fitness or child.fitness == parents[1].fitness:
                child.mutate(self.mutation_probability)
                child.updateFitness()

        # Delete the worst, and free the memory
        to_remove = int(np.floor(self.n_population * self.epuration_factor))
        for i in range(to_remove):
            elem = gen.pop()
            del elem

        # Add the children into the population and sort
        self.population = gen + list(children)
        self.sort()

        genlen = len(self.population)
        for i in range(genlen):
            
            # Skip the elite, we don't want to get bad results after getting a good one
            if i == 0:
                continue
            
            # Mutate with the appropriate probability
            if np.random.random() >= 0.6 - (i / genlen) * (i / genlen):
                self.population[i].mutate(self.mutation_probability)

        print("mutated: {0:}/{1:}, {2:.2f}% ".format(i, genlen, i / genlen * 100))

        # Now we need to create as much as needed new individuals
        # Introducing new random individuals is the best way to not be trapped by the evolution
        self.create()

    def evolve(self, epochs=1, callback=None):
        '''
            1. Evaluation of the fitness of each chromosome and sorting
            2. Select 2 good-rated parents for next generation
            3. Generate next population
        '''

        for epoch in range(epochs):

            print("{} -------------------------- ".format(epoch))

            self.evaluate()
            selection = self.select();
            self.generate(selection)

            if callback is not None:
                callback()
                sys.stdout.flush()

            sys.stdout.flush()

        return self.population

    def visual_analysis(self):

        history_size = len(list(self.skeleton_stats.items())[0][1])
        colors = np.random.rand(history_size, 3)

        for indicator, stats in self.skeleton_stats.items():

            stats.sort(key=lambda props: props[0]) # props[0] == score
            stats = np.asarray(stats[::-1]) # rank by larger to smaller score

            fig = plt.figure(figsize=(8, 4))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax1 = plt.subplot(gs[0])

            for i, (score, indicator_value) in enumerate(stats):
                score = float(score)
                # Hide really bad individuals from the graphs
                if score >= _DEFAULT_FITNESS:
                    continue

                m = (5, 1, 0) if score == self.elite.fitness else "." # start shape if it is the elite
                c = (1, 0, 0) if score == self.elite.fitness else colors[i]

                ax1.semilogx([score], [indicator_value], marker=m, markersize=i / stats.shape[0] * 15 + 10, color=c)

            ax2 = plt.subplot(gs[1])
            ax2.hist(stats[:, -1], orientation="horizontal")  # hist on values

            plt.suptitle(indicator, fontsize=18)
            ax1.set_xlabel("Fitness score")
            ax1.set_ylabel("Value")
            plt.tight_layout()
            plt.show()

# Theses three cannot be pickled by the async mapping as class methods
# They need to be top-level function of the module. It's a bit ugly, but it's easy

def update_individual(elem):
    elem.updateFitness()

def display_progress(result=None):
    self.count += 1
    print("\revaluation: {2:.2%}\t({0:} over {1:})".format(self.count, self.n_population, self.count/self.n_population), end="")

def display_error(arg):
    print("An error occured", arg)
