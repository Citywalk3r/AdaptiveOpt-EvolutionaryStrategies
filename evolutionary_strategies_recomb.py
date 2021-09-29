import itertools
import random
import math
import numpy as np
from queue import Queue

class Evolutionary_Strategies:

    def __init__(self, n=2, eval_func=None, init_sigma=0.05, l=6, m=1):
        
        self.n = n
        self.eval_func = eval_func
        self.sigma = init_sigma
        self.l = l
        self.m = m
        self.best_so_far = 999999999999999
        self.mutations_performed = 0
        self.sigma_list = []


        # Initialize a queue to keep last 10*n mutations
        self.q = Queue(maxsize = 10*n)
    
    def generate_init_pop(self, seed, initial_sigma):
        """
        Generates the initial population given population size m.
        """
        rng = np.random.default_rng(seed)
        init_generation = [[*np.ndarray.tolist(rng.uniform(-32,32,2)), *[initial_sigma, initial_sigma]] for _ in range(self.m)]
        # print(init_generation)
        return init_generation

    def calculate1fifth(self):
        """
        Calculates the success rate of children over parent
        """
        truth_table = np.asarray(list(self.q.queue))
        successful_children = np.count_nonzero(truth_table)
        success_rate = successful_children/truth_table.size
        # print("Success rate: ", success_rate)
        # print("Total number of children that are better than parent in the last {:} mutations: {}".format(10*self.n, successful_children))

        return success_rate

    def mutation_sigma_recombination(self, parent, pop):
            """Implements the mutation operator.
            Normally distributed pertrubation, 𝑁(0,𝜎) with the same 𝜎 for each encoded variable
            
            Parameters:
                parent : 2D array [x,y]
            Returns:
                children : list of children arrays [[x1,y1], [x2,y2] ... [x l//m,y l//m]]

            """
            children = []
            parent_score = self.eval_func(parent)

            for _ in range(self.l//self.m):

                parents_for_recomb = random.sample(pop, 4)
                avg_parent_sigma_x = (parents_for_recomb[0][2] + parents_for_recomb[1][2])/2
                avg_parent_sigma_y = (parents_for_recomb[2][3] + parents_for_recomb[3][3])/2
                
               
                child = parent.copy()
                # print(self.q.qsize())

                if(self.mutations_performed%self.n == 0 and self.q.full()):
                    success = self.calculate1fifth()
                    if success > 0.2:
                        child[2] = avg_parent_sigma_x/0.85
                        child[3] = avg_parent_sigma_y/0.85
                    elif success < 0.2:
                        child[2] = 0.85*avg_parent_sigma_x
                        child[3] = 0.85*avg_parent_sigma_y
                    # print("New sigma: ", self.sigma) # Does the new sigma need to be withing bounds?
                    # self.sigma_x_list.append(child[2])

                modifier_x = np.random.normal(loc=0.0, scale=child[2])
                modifier_y = np.random.normal(loc=0.0, scale=child[3])
                # print(self.best_so_far)
                # print("Modifier: ", modifier)

            
                if child[0] + modifier_x <=32 and child[0] + modifier_x >=-32: # Check if out of boundaries
                    child[0] += modifier_x
                    

                if child[1] + modifier_y <=32 and child[1] + modifier_y >=-32: # Check if out of boundaries
                    child[1] += modifier_y
                
                if self.q.full():
                    # Removing element from queue
                    # print("\nElement dequeued from the queue")
                    self.q.get()

                self.q.put(self.eval_func(child) < parent_score)

                children.append(child.copy())

                self.mutations_performed+=1
    
            return children

    def select_m_best(self, pop):
        """
        Sorts the list based on the evaluation function and returns the first m elements.
        """
        m_best = sorted(pop, key=lambda child: self.eval_func(child))[:self.m]
        return m_best

    def es(self, m=5, l=30, g_max=5, eval_f= None, seed=0, initial_sigma=0.05, recombination_f=None, strategy="k+l", n=2):

            """Evolutionary Strategies

            Parameters:
                ipop_f : initial population generation function
                m : μ
                l : λ ~ Usually λ≫μ (high selection pressure, typically 6 to 1 or greater) 
                g_max : max number of generations
                eval_f : evaluation function
                mutation_f: muation function
                initial_sigma: initial standard deviation for normally distributed perturbation
                recombination_f : recombination function
                strategy: string "k+l" or "k,l"
                n: number of mutations every which to adapt sigma

            Returns:
                x_final : final state array
                generation_list : list of historical states

            """
            
            print("Running the evolutionary strategies algorithm...")

            # region initialization
            self.n = n
            self.eval_func = eval_f
            self.l = l
            self.m = m

            # final populations for each generation (used for visualization)
            generation_list = []
            g = 0
            initial_population = self.generate_init_pop(seed, initial_sigma)
            # generation_list.append(initial_population)

            # endregion
            pop = initial_population
            # print("\nInitial population ", pop)
            while (g<g_max):
                
                
                parents = random.sample(pop, self.m)
                # print ("\nSampled parents: ", parents)
                if(strategy == "k,l"):
                    pop = []
                for parent in parents:

                    children = self.mutation_sigma_recombination(parent, pop)
                    # print("Children from 1 parent: ", children)
                    # print("l/m: ", l//m)
                    # print("Length of children: ", len(children))
                    pop.extend(children)
                    # print("Total population with children: ", pop)
                    # print("Length of total population: ", len(pop))
                    # print("Total population with children: ", pop)
                    # print("Length of total population: ", len(pop))
                pop = self.select_m_best(pop)
                # print(pop)
                # if self.eval_func(pop[0])< self.best_so_far:
                #     self.best_so_far = self.eval_func(pop[0])
                # print("Population length after selection of best: ", len(pop))
                generation_list.append(list(pop))
                g+=1
            return generation_list, self.sigma_list