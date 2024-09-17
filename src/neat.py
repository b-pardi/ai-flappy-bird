import numpy as np
from enum import Enum
from src.components import Bird
from tabulate import tabulate
import logging

class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Net:
    def __init__(self, n_in_nodes=3, n_out_nodes=1, n_bias_nodes=1, n_init_conns='FC'):
        self.node_dtype = np.dtype([ 
            ('id', np.int32),
            ('type', np.int8), # corresponds to enum values
            ('input_sum', np.float32),
            ('output_value', np.float32),
        ])

        self.connection_dtype = np.dtype([
            ('in_node_id', np.int32),
            ('out_node_id', np.int32),
            ('weight', np.float32),
            ('enabled', np.bool),
            ('innovation', np.int32),
        ])

        if n_init_conns == 'FC':
            is_fully_connected = True
            n_init_conns = (n_in_nodes + n_bias_nodes) * n_out_nodes
        else:
            is_fully_connected = False

        self.nodes = np.zeros(n_in_nodes + n_out_nodes + n_bias_nodes, dtype=self.node_dtype)
        self.connections = np.zeros(n_init_conns, dtype=self.connection_dtype)

        # initalize network to have 3 input nodes (and 1 bias node in the input layer) fully connected to 1 output node
        for i in range(n_in_nodes):
            self.nodes[i] = (i, NodeType.INPUT.value, 0.0, 0.0)

        for i in range(n_in_nodes, n_in_nodes + n_out_nodes):
            self.nodes[i] = (i, NodeType.OUTPUT.value, 0.0, 0.0)

        # bias node
        self.bias_idx = n_in_nodes + n_out_nodes
        self.nodes[self.bias_idx] = (self.bias_idx, NodeType.INPUT.value, 0.0, np.random.rand(1))
            
        in_node_idxs = np.where(self.nodes['type'] == NodeType.INPUT.value)[0]
        out_node_idxs = np.where(self.nodes['type'] == NodeType.OUTPUT.value)[0]

        if is_fully_connected:
            conn_idx = 0
            for in_node_idx in in_node_idxs:
                for out_node_idx in out_node_idxs:
                    init_weight = np.random.uniform(-1,1)
                    self.connections[conn_idx] = (in_node_idx, out_node_idx, init_weight, True, conn_idx)
                    conn_idx += 1
        
        elif n_init_conns > 0:
            conn_idx = 0
            while conn_idx < n_init_conns:
                in_node_idx = np.random.choice(in_node_idxs)
                out_node_idx = np.random.choice(out_node_idxs)
                init_weight = np.random.uniform(-1,1)
                self.connections[conn_idx] = (in_node_idx, out_node_idx, init_weight, True, conn_idx)
                conn_idx += 1

    def add_node(self, id):
        new_node = np.array([(
                id,                     # new node id is just incr curr greatest id
                NodeType.HIDDEN.value,  # mutated nodes will always be hidden nodes
                0.0,                    # initial input value 0
                0.0                     # initial output value 0
            )], dtype=self.node_dtype)

        self.nodes = np.append(self.nodes, new_node)

    def add_connection(self, in_node_id, out_node_id, weight, innovation):
        new_conn = np.array([(
            in_node_id, # new node connects feeds into this second new conn
            out_node_id,# new conn output node is original conns output node
            weight,     # weight from original connection
            True,       # new conn enabled
            innovation  # new innovation number
        )], dtype=self.connection_dtype)

        self.connections = np.append(self.connections, new_conn)

    def get_connected_nodes(self):
        enabled_conns = self.connections[self.connections['enabled']]
        connected_node_ids = np.unique(np.concatenate((enabled_conns['in_node_id'], enabled_conns['out_node_id'])))
        connected_nodes = self.nodes[(np.isin(self.nodes['id'], connected_node_ids)) | (self.nodes['type'] == NodeType.INPUT.value)]
        return connected_nodes

    def mutate_weight(self):
        for connection in self.connections:
            if np.random.rand() < 0.1:  # 10% chance to completely change the weight
                connection['weight'] = np.random.normal()
            else:  # 90% chance to slightly adjust the weight
                connection['weight'] += np.random.normal() * 0.1  # Adjust by small amount

    def mutate_bias(self, common_mutation_power=0.5):
        if np.random.rand() < 0.1:
            self.nodes['output_value'][self.bias_idx] += np.random.normal()
        else:
            self.nodes['output_value'][self.bias_idx] += np.random.normal(-common_mutation_power, common_mutation_power)

    def add_node_mutation(self):
        # randomly choose connection to disable and split
        enabled_conn_idxs = np.where(self.connections['enabled'])[0]
        if len(enabled_conn_idxs) == 0:
            return
        
        chosen_conn_idx = np.random.choice(enabled_conn_idxs)
        chosen_conn = self.connections[chosen_conn_idx]

        # disable randomly chosen connection
        #self.connections[chosen_conn_idx]['enabled'] = False

        # make new node
        new_node_id = max(self.nodes['id']) + 1
        self.add_node(new_node_id)

        '''self.add_connection( # new conn 1
            chosen_conn['in_node_id'],              # original in node id now connected to this new connection
            new_node_id,                            # new node is now output of this new conn
            1.0,                                    # initial weight of 1.0 so new node does not drastically change behavior, since new_conn2 preserves old_conn's weight
            max(self.connections['innovation']) + 1 # new innovation number
        )

        self.add_connection( # new conn 2
            new_node_id,                            # new node connects feeds into this second new conn
            chosen_conn['out_node_id'],             # new conn output node is original conns output node
            chosen_conn['weight'],                  # weight from original connection
            max(self.connections['innovation']) + 1 # new innovation number
        )'''

    def add_connection_mutation(self):
        """
        Mutate the genome by adding a new connection between two nodes that are not currently connected.
        Ensures the new connection does not create cycles in the feedforward network.
        """
        potential_node_pairs = []
        for n1 in self.nodes:
            for n2 in self.nodes:
                if self.valid_pair(n1, n2):
                    potential_node_pairs.append((n1, n2))

        if not potential_node_pairs:
            return
        
        potential_node_pairs = np.array(potential_node_pairs, dtype=object) # dtype object for tuples
        choice_idx = np.random.randint(len(potential_node_pairs))
        chosen_pair = potential_node_pairs[choice_idx]

        self.add_connection(
            chosen_pair[0]['id'],   # connect first node of pair as in node
            chosen_pair[1]['id'],   # output of new conn is second node of random pair
            np.random.uniform(-1,1),                    # init weight random since new conn is supposed allow net to explore new behaviors
            max(self.connections['innovation']) + 1
        )

    def valid_pair(self, in_node, out_node):
        """
        check if node pair is valid before inserting new one
        checks following conditions:
            - in and out nodes are the same
            - output node of connection is a type less than or equal to input node
                - i.e. if it would connect an input node to an input node or an output node back to a hidden node
            - if connection already exists
            - if it forms a cycle

        returns false if any are true
        """
        if in_node['id'] == out_node['id'] or out_node['type'] <= in_node['type']:
            return False
        
        for conn in self.connections:
            if conn['in_node_id'] == in_node['id'] and \
            conn['out_node_id'] == out_node['id']  and \
            conn['enabled']:
                
                return False
            
        if self.is_cyclic(in_node['id'], out_node['id'] ):
            return False
        
        return True
    
    def is_cyclic(self, in_node_id, out_node_id):
        """
        check if adding conn between 2 nodes would create a cycle
        use dfs starting from the out_node and check if it ever gets back to the in node
        """

        stack = [out_node_id]
        visited = set()

        while stack: # while any node is on the stack, nodes get added to stack during traversal
            cur_node_id = stack.pop()
            if cur_node_id == in_node_id: # this indicates there is a path from out node back to start, meaning it's cyclic
                return True
            
            visited.add(cur_node_id)

            # grab all connections that are starting at this current node
            cur_connections = self.connections[self.connections['in_node_id'] == cur_node_id]
            for conn in cur_connections:
                if conn['enabled']:
                    next_node_id = conn['out_node_id'] # next node is the next connected node to this cur nodes cur connection
                    if next_node_id not in visited:
                        stack.append(next_node_id)

        return False # if while loop finishes without returning True then no cycles found

    @staticmethod
    def activate(input_sum): 
        # sigmoid activation
        return 1 / (1 + np.exp(-input_sum))
        # ReLU activation
        #return np.maximum(0, input_sum)

    def feed_forward(self, inputs):
        # grab input node indices
        input_node_idxs = self.nodes['type'] == NodeType.INPUT.value # id input nodes
        bias = self.nodes['output_value'][self.bias_idx] # bias is an input but set it separately
        self.nodes['input_sum'][input_node_idxs] = inputs + [bias] # set given input values for input nodes

        # inputs don't get activated, so they get sent straight to output_value
        self.nodes['output_value'][input_node_idxs] = self.nodes['input_sum'][input_node_idxs]

        # clear input sums before forward pass
        non_input_node_idxs = self.nodes['type'] != NodeType.INPUT.value
        self.nodes['input_sum'][non_input_node_idxs] = 0

        # prop through the network
        for conn in self.connections:
            if conn['enabled']:
                # find idx of out node (0 idx cuz np where returns tuple of arrs even though there is only one arr in tuple)
                conn_output_node_idx = np.where(self.nodes['id'] == conn['out_node_id'])[0]
                conn_input_node_idx = np.where(self.nodes['id'] == conn['in_node_id'])[0]
                
                if conn_input_node_idx:
                    in_val = self.nodes['output_value'][conn_input_node_idx][0]
                    self.nodes['input_sum'][conn_output_node_idx] += in_val * conn['weight']

        # activate non input nodes        
        self.nodes['output_value'][non_input_node_idxs] = self.activate(self.nodes['input_sum'][non_input_node_idxs])
            
        # get and return outputs
        output_node_idxs = self.nodes['type'] == NodeType.OUTPUT.value
        return self.nodes['output_value'][output_node_idxs]
    
class Population:
    def __init__(
            self,
            size,
            n_in_nodes=3,
            n_out_nodes=1,
            n_bias_nodes=1,
            n_init_conns='FC',
            compatibility_thresh=3.0,
            excess_gene_coeff=1.0,
            disjoint_gene_coeff=0.7,
            weight_coeff=0.5
        ):
        self.size = size
        self.birds = np.empty(size, dtype=object)
        self.gen = 1
        self.compatibility_thresh = compatibility_thresh
        self.excess_gene_coeff = excess_gene_coeff
        self.disjoint_gene_coeff = disjoint_gene_coeff
        self.weight_coeff = weight_coeff

        # set up logggers
        self.stats_logger = self.init_logger('fitness_stats_log', 'logs/fitness_stats.log')
        self.net_logger = self.init_logger('networks_log', 'logs/networks.log')

        for i in range(self.size):
            net = Net(n_in_nodes, n_out_nodes, n_bias_nodes, n_init_conns)
            self.birds[i] = Bird(net)

    def evaluate_fitness(self):
        highest_fitness = -1
        for bird in self.birds:
            bird.evaluate_fitness()
            if bird.fitness > highest_fitness:
                highest_fitness = bird.fitness
                fittest_bird = bird
        
        return fittest_bird

    def init_logger(self, name, log_file):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(log_file, 'w')
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

    def compute_compatibility_distance(self, net1, net2):
        # only considering enabled conns
        conns1 = net1.connections[net1.connections['enabled']]
        conns2 = net2.connections[net2.connections['enabled']]

        # edge case if no enabled conns
        if len(conns1) == 0 or len(conns2) == 0:
            return float('inf')

        # get relevant information from nets
        sorted_idxs1 = np.argsort(conns1['innovation'])
        sorted_idxs2 = np.argsort(conns2['innovation'])
        net1_innovations = conns1['innovation'][sorted_idxs1]
        net1_weight = conns1['weight'][sorted_idxs1]
        net2_innovations = conns2['innovation'][sorted_idxs2]
        net2_weight = conns2['weight'][sorted_idxs2]

        # find maximum innovation
        max_innovation1 = np.max(net1_innovations) if len(net1_innovations) > 0 else 0
        max_innovation2 = np.max(net2_innovations) if len(net2_innovations) > 0 else 0

        # find matching genes
        matching_innovations = np.intersect1d(net1_innovations, net2_innovations, assume_unique=True)

        # find weight diffs for matching genes
        if len(matching_innovations) > 0:
            # find indices of matching innovations in both genomes
            indices1 = np.searchsorted(net1_innovations, matching_innovations)
            indices2 = np.searchsorted(net2_innovations, matching_innovations)
            
            # Get the corresponding weights for matching innovations
            matching_weights1 = net1_weight[indices1]
            matching_weights2 = net2_weight[indices2]
            
            # Calculate absolute weight differences
            weight_diffs = np.abs(matching_weights1 - matching_weights2)
            average_weight_diff = np.mean(weight_diffs)
        else:
            average_weight_diff = 0

        # find excess genes
        # genes in the net with more genes that aren't in the other, past the smaller net's largest innovation num
        if max_innovation1 > max_innovation2:
            excess_genes = np.sum(net1_innovations > max_innovation2) # summing boolean values
        else:
            excess_genes = np.sum(net2_innovations > max_innovation1) # summing boolean values

        # find disjoint genes
        # genes in one net that aren't in the other within range of innvation nums of the other
        disjoint_genes1 = np.sum(~np.isin(net1_innovations, net2_innovations) & (net1_innovations <= max_innovation2))
        disjoint_genes2 = np.sum(~np.isin(net2_innovations, net1_innovations) & (net2_innovations <= max_innovation1))

        # normalizing factor (normalize by greatest num of innovations)
        N = max(len(net1_innovations), len(net2_innovations))
        if N < 15:
            N = 1 # don't normalize on small nets, it may over normalize

        compatibility_distance = (self.excess_gene_coeff * excess_genes / N) +\
                                (self.disjoint_gene_coeff * (disjoint_genes1 + disjoint_genes2) / N) +\
                                (self.weight_coeff * average_weight_diff)
        
        return compatibility_distance

    def speciation(self):
        """
        Group the population of birds into species based on compatibility distance.
        Each bird is compared with the representatives of existing species to determine which species it belongs to.
        If a bird doesn't match any existing species, it forms a new species.
        """
        self.species = [] # will be a nested list, where each sublist contains birds within the species
        self.species_reps = [] # holds birds that represent a species for comparison

        for bird in self.birds:
            found_species = False
            
            # compare species representative bird to existing species representatives
            for i, rep in enumerate(self.species_reps):
                distance = self.compute_compatibility_distance(bird.net, rep.net)

                if distance < self.compatibility_thresh: # if bird close enough to rep bird
                    self.species[i].append(bird)
                    found_species = True
                    break
            
            if not found_species: # if pre-existing species not found, make new one
                self.species_reps.append(bird)
                self.species.append([bird]) # start new list for this species of birds

        # update species reps so the best bird of the species is the rep
        for i, species in enumerate(self.species):
            best_bird = max(species, key=lambda bird:bird.fitness)
            self.species_reps[i] = best_bird
        
    def select_parents(self):
        """
        Select parents from the population for crossover.
        use fitness-proportionate selection.
        """
        fitness_scores = np.array([bird.fitness for bird in self.birds], dtype=float)
        fitness_sum = np.sum(fitness_scores)

        # edge case if all scores 0
        if fitness_sum == 0:
            # randomly choose parents
            parent_idxs = np.random.choice(self.birds.shape[0], size=self.size, replace=True)
            return self.birds[parent_idxs]
        
        # normalize fitness scores so we can make a probability distribution where larger scores equate to higher selection chance
        norm_fitness_scores = fitness_scores / fitness_sum
        parent_idxs = np.random.choice(self.birds.shape[0], size=self.size, replace=True, p=norm_fitness_scores)
        return self.birds[parent_idxs]

    def crossover(self, parent1, parent2):
        """
        Combine the genomes of two parents to produce an offspring.
        Handle matching genes, disjoint genes, and excess genes based on fitness and innovation numbers.
        """
        if parent2.fitness > parent1.fitness:
            # ensure parent 1 fitness greater, since crossover favors fitter parent's genes
            # if both parents are equally fit, it doesn't matter which parent we take from, so we just take from parent 1
            parent1, parent2 = parent2, parent1

        conns1_enabled_idxs = parent1.net.connections['enabled']
        conns1 = parent1.net.connections[conns1_enabled_idxs]
        conns2_enabled_idxs = parent2.net.connections['enabled']
        conns2 = parent2.net.connections[conns2_enabled_idxs]

        # offspring will have same number of conns as fitter or equally fit parent (parent1/conns1)
        offspring_conns = np.zeros(len(conns1), dtype=parent1.net.connection_dtype)
        offspring_count = 0

        # find matching innovations
        matching_innovations = np.intersect1d(conns1['innovation'], conns2['innovation'], assume_unique=True)

        # crossover mathcing genes
        for innovation in matching_innovations:
            # randomly choose matching gene from each parent
            if np.random.rand() >= 0.5:
                # 50% chance we choose matching gene from parent1
                cur_matching_idx_parent1 = np.where(conns1['innovation'] == innovation)[0][0]
                offspring_conns[offspring_count] = conns1[cur_matching_idx_parent1]
            else:
                # 50% chance choose matching gene from parent2
                cur_matching_idx_parent2 = np.where(conns2['innovation'] == innovation)[0][0]
                offspring_conns[offspring_count] = conns2[cur_matching_idx_parent2]

            offspring_count += 1

        # disjoint/excess genes from parent 1 (fitter or equally fit parent)
        # distinction between excess and disjoint not relevant for crossover,
        # so we can refer to either disjoint or excess as mismatching
        mismatching_innovations = conns1['innovation'][~np.isin(conns1['innovation'], conns2['innovation'])]
        for innovation in mismatching_innovations:
            # THIS SHOULD BE A MAPPING OUTSIDE THE LOOP LATER FOR EFFICIENCY
            cur_mismatching_idx = np.where(conns1['innovation'] == innovation)[0][0]
            offspring_conns[offspring_count] = conns1[cur_mismatching_idx]
            offspring_count += 1

        # init offsprings network with no conns, so that we can use cross over conns only
        child_net = Net(n_init_conns=0)

        # copy hidden nodes from parent 1, 
        # since connections will be either only to nodes parent 1 had,
        # or to nodes both parent 1 and 2 had
        hidden_nodes = parent1.net.nodes[parent1.net.nodes['type'] == NodeType.HIDDEN.value]
        for node in hidden_nodes:
            child_net.add_node(node['id'])

        # add connections found from crossover
        for conn in offspring_conns:
            child_net.add_connection(
                conn['in_node_id'],
                conn['out_node_id'],
                conn['weight'],
                conn['innovation']
            )
        
        return Bird(child_net)

    def mutate(self, weight_mutation_weight=0.8, bias_mutation_rate=0.7, add_node_mutation_rate=0.05, add_connection_mutation_rate=0.12):
        """
        Apply mutations to a bird's network. This could include adding nodes, adding connections, or perturbing weights.
        """
        for bird in self.birds:
            if np.random.rand() <= weight_mutation_weight:
                bird.net.mutate_weight()
            if np.random.rand() <= bias_mutation_rate:
                bird.net.mutate_bias()
            if np.random.rand() <= add_node_mutation_rate:
                bird.net.add_node_mutation()
            if np.random.rand() <= add_connection_mutation_rate:
                bird.net.add_connection_mutation()
            

    def repopulate(self):
        """
        Create a new generation of birds by selecting parents, applying crossover, and mutating the offspring.
        """
        new_gen = np.empty(self.size, dtype=object)
        parents = self.select_parents()

        for i in range(self.size):
            # randomly choose 2 unique parents and create child via crossover
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            offspring = self.crossover(parent1, parent2)
            new_gen[i] = offspring

        # mutate offspring
        self.birds = new_gen
        self.mutate()

        self.gen += 1

    def log_progress(self):
        """
        Log the progress of the evolution, including fitness scores, species stats, and other metrics.
        """
        best_fitness = max(bird.fitness for bird in self.birds)
        worst_fitness = min(bird.fitness for bird in self.birds)
        avg_fitness = np.mean([bird.fitness for bird in self.birds])
        n_species = len(self.species)
        print(f"Generation {self.gen} Global Stats:\nBest Fitness = {best_fitness}, Worst Fitness = {worst_fitness}, Average Fitness = {avg_fitness}, Num Species: {n_species}")
        print("Individual Species Top Stats:")
        headers = [
            'Species Size',
            'Best Fitness',
            'Worst Fitness',
            'Avg Fitness',
            'Most Pipes Passed',
            'Most Nodes',
            'Most Conns'
        ]

        species_stats = []
        for i, species in enumerate(self.species):
            species_size = len(species)
            best_fitness = max(bird.fitness for bird in species)
            worst_fitness = min(bird.fitness for bird in species)
            avg_fitness = np.mean([bird.fitness for bird in species])
            most_pipes = max(bird.pipes_passed for bird in species)
            most_nodes = max(len(bird.net.nodes) for bird in species)
            most_conns = max(len(bird.net.connections) for bird in species)
            
            species_stats.append([
                species_size,
                best_fitness,
                worst_fitness,
                avg_fitness,
                most_pipes,
                most_nodes,
                most_conns
            ])

            log_string = f"Generation: {self.gen} Species {i+1}/{n_species}\n"
            for j, header in enumerate(headers):
                log_string += f"{header}: {species_stats[i][j]}, "
            self.stats_logger.info(f"{log_string}\n")

        print(tabulate(species_stats, headers, tablefmt='grid'))

        # log best bird architecture
        best_bird = max(self.birds, key=lambda bird: bird.fitness)
        node_dicts, conn_dicts = [], []
        for node in best_bird.net.nodes:
            node_dicts.append({
                'id': int(node['id']),
                'type': int(node['type']),
                'input_sum': float(node['input_sum']),
                'output_value': float(node['output_value'])
            })
        for conn in best_bird.net.connections:
            conn_dicts.append({
                'in_node_id': int(conn['in_node_id']),
                'out_node_id': int(conn['out_node_id']),
                'weight': float(conn['weight']),
                'enabled': bool(conn['enabled']),
                'innovation': int(conn['innovation'])
            })

        self.net_logger.info(f"Generation {self.gen} Best Bird:")
        self.net_logger.info(f"Nodes: {node_dicts}")
        self.net_logger.info(f"Connections: {conn_dicts}\n")    