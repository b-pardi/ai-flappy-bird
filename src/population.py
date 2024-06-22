import operator
import numpy as np
import random

import src.components as comps

class Population:
    def __init__(self, size):
        self.size = size
        self.birds = [comps.Bird() for _ in range(self.size)]
        self.generation = 1
        self.species = []

    def update_live_players(self):
        for bird in self.birds:
            if bird.alive:
                bird.look()
                bird.think()
                bird.draw_bird(comps.window)
                bird.update(comps.ground)

    def natural_selection(self):
        print("Speciation starting")
        self.speciate()

        print("Finding fitness..")
        self.calculate_fitness()
        self.kill_extinct()
        self.kill_stale()
        self.sort_species_by_fitness()

        print("Spawning next generation")
        self.next_gen()

    def speciate(self):
        # clear existing species list
        for s in self.species:
            s.birds = []

        # add birds to species if they are similar enough to existing species
        for bird in self.birds:
            add_to_species = False
            for s in self.species:
                if s.similarity(bird.brain):
                    s.add_to_species(bird)
                    add_to_species = True
                    break

            # if birds not similar enough to existing species make new species
            if not add_to_species:
                self.species.append(Species(bird))

    def calculate_fitness(self):
        for bird in self.birds:
            bird.calculate_fitness()
        for s in self.species:
            s.calculate_avg_fitness()

    def kill_extinct(self):
        species_bin = []
        for s in self.species:
            if len(s.birds) < 3:
                species_bin.append(s)
        for s in species_bin:
            self.species.remove(s)

    def kill_stale(self):
        bird_bin = []
        species_bin = set()
        max_fit = 0
        for s in self.species:
            if s.average_fitness > max_fit:
                max_fit = s.average_fitness

        print(max_fit, s.average_fitness)
        for s in self.species:
            if s.average_fitness < max_fit / 2:
                species_bin.add(s)

        for s in self.species:
            if s.staleness >= 5: # 8 generations of not winning to remove
                if len(self.species) >= len(species_bin):
                    species_bin.add(s)
                    for bird in s.birds:
                        bird_bin.append(bird)
                else:
                    s.staleness = 0

        for bird in bird_bin:
            self.birds.remove(bird)
        for s in species_bin:
            self.species.remove(s)

    def sort_species_by_fitness(self):
        for s in self.species:
            s.sort_birds_by_fitness()
        
        self.species.sort(key=operator.attrgetter('benchmark_fitness'), reverse=True)

    def next_gen(self):
        # kill existing children
        children = []

        # clone of champion added to each species
        for s in self.species:
            print(s.champion.fitness)
            children.append(s.champion.clone())

        # fill open bird slots with children (mutated)
        children_per_species = int((self.size - len(self.species)) / len(self.species))
        for s in self.species:
            for i in range(children_per_species):
                children.append(s.offspring())

        while len(children) < self.size:
            children.append(self.species[0].offspring())

        self.birds = []
        for child in children:
            self.birds.append(child)

        print(f"Generation {self.generation} ending, starting next...")
        self.generation += 1

    def is_extinct(self):
        extinct = True
        for bird in self.birds:
            if bird.alive:
                extinct = False
        return extinct
    
class Species:
    def __init__(self, bird):
        self.birds = []
        self.birds.append(bird)
        self.average_fitness = 0
        self.benchmark_fitness = bird.fitness
        self.benchmark_brain = bird.brain.clone()
        self.champion = bird.clone()
        self.thresh = 1.2
        self.staleness = 0

    def similarity(self, brain):
        similarity = self.weight_diff(self.benchmark_brain, brain)
        return self.thresh > similarity
    
    @staticmethod
    def weight_diff(brain1, brain2):
        total_weight_diff = 0
        
        for i in range(len(brain1.connections)):
            for j in range(len(brain2.connections)):
                if i == j: # looking at equiv connections in brain
                    total_weight_diff += abs(brain1.connections[i].weight - brain2.connections[j].weight)
        return total_weight_diff
    
    def add_to_species(self, bird):
        self.birds.append(bird)

    def sort_birds_by_fitness(self):
        self.birds.sort(key=operator.attrgetter('fitness'), reverse=True)

        if self.birds[0].fitness > self.benchmark_fitness:
            self.staleness = 0 # reset staleness if a species member becomes champion
            self.benchmark_fitness = self.birds[0].fitness
            self.champion = self.birds[0].clone()
        else:
            self.staleness += 1

    def calculate_avg_fitness(self):
        total_fitness = 0
        for bird in self.birds:
            total_fitness += bird.fitness

        if self.birds:
            self.average_fitness = int(total_fitness / len(self.birds))
        else:
            self.average_fitness = 0

    def offspring(self):
        child = self.birds[random.randint(1, len(self.birds)) - 1].clone()
        child.brain.mutate()
        return child
