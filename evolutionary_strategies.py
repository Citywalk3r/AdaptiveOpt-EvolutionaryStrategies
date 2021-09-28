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


        # Initialize a queue to keep last 10*n mutations
        self.q = Queue(maxsize = 10*n)
    
    def generate_init_pop(self, seed):
        """
        Generates the initial population given population size m.
        """
        rng = np.random.default_rng(seed)
        init_generation = [np.ndarray.tolist(rng.uniform(-32,32,2)) for _ in range(self.m)]
        # print(init_generation)
        return init_generation

    def calculate1fifth(self, parent):
        # calculate the parent score
        parent_score = self.eval_func(parent)

        # get the elements of the queue to a list
        children = list(self.q.queue)

        # generate the list of children scores
        children_scores = [self.eval_func(child) for child in children]
        # print(children_scores)

        # convert to numpy array
        children_scores_np = np.asarray(children_scores)

        # get the number of successful children
        num_successful_children = (children_scores_np > parent_score).sum()
        # print("Successful children number: ", num_successful_children)
        # print("Length of children ",len(children))

        # divide the number of successful children by total children
        success_rate = num_successful_children / len(children)
        print("Success rate: ", success_rate)

        return success_rate

    def mutation_same_std_dev(self, parent):
            """Implements the mutation operator.
            Normally distributed pertrubation, 𝑁(0,𝜎) with the same 𝜎 for each encoded variable
            
            Parameters:
                parent : 2D array [x,y]
                sigma : initial population generation function
                m : μ
                l : λ ~ Usually λ≫μ (high selection pressure, typically 6 to 1 or greater) 
                n : mutation number after which to attempt std deviation modification based on the 1/5 rule.
            Returns:
                children : list of children arrays [[x1,y1], [x2,y2] ... [x l//m,y l//m]]

            """

            # print("Parent:", parent)

            children = []
            mutations_so_far = 0

            

            # print("l/m: ", l//m)
            for _ in range(self.l//self.m):
                if(mutations_so_far == self.n and self.q.qsize()==10*self.n ):
                    mutations_so_far=0
                    success = self.calculate1fifth(parent)
                    if success > 0.2:
                        self.sigma = self.sigma/0.85
                    elif success < 0.2:
                        self.sigma = 0.85*self.sigma

                child = parent.copy()
                modifier = np.random.normal(loc=0.0, scale=self.sigma)
                print("sigma: ", self.sigma)
                # print("Modifier: ", modifier)

                
            
                if child[0] + modifier <=32 and child[0] + modifier >=-32: # Check if out of boundaries
                    child[0] += modifier
                # else:
                #     child = np.ndarray.tolist(np.random.uniform(-32,32,2))
                
                if child[1] + modifier <=32 and child[1] + modifier >=-32: # Check if out of boundaries
                    child[1] += modifier
                # else:
                #     child = np.ndarray.tolist(np.random.uniform(-32,32,2))
                # print("Child x {:}, child y {:}".format(child[0],child[1]))
            

                children.append(child)
                # print("children so far: ", children)

                if self.q.full():
                    # Removing element from queue
                    # print("\nElement dequeued from the queue")
                    self.q.get()
                self.q.put(child)
                # print(self.q.qsize())

                mutations_so_far += 1

            return children

    def select_m_best(self, pop):
        """
        Sorts the list based on the evaluation function and returns the first m elements.
        """
        m_best = sorted(pop, key=lambda child: self.eval_func(child))[:self.m]
        return m_best

    def es(self, m=5, l=30, g_max=5, eval_f= None, seed=0, initial_sigma=0.05, recombination_f=None, strategy="k+l", n=5):

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
            self.sigma = initial_sigma
            self.l = l
            self.m = m

            # final populations for each generation (used for visualization)
            generation_list = []
            g = 1
            initial_population = self.generate_init_pop(seed)
            # generation_list.append(initial_population)

            # endregion
            pop = initial_population
            print("\nInitial population ", pop)
            while (g<=g_max):
                
                parents = random.sample(pop, self.m)
                # print ("\nSampled parents: ", parents)
                if(strategy == "k,l"):
                    pop = []
                for parent in parents:
                    children = self.mutation_same_std_dev(parent)
                    # print("Children from 1 parent: ", children)
                    # print("l/m: ", l//m)
                    # print("Length of children: ", len(children))
                    pop.extend(children)
                    # print("Total population with children: ", pop)
                    # print("Length of total population: ", len(pop))
                    
                    # print("Generation: ",g)
                    # print("Population size: ",len(generation))
                    
                    # print("Total population with children: ", pop)
                    # print("Length of total population: ", len(pop))
                pop = self.select_m_best(pop)
                # print("Population length after selection of best: ", len(pop))
                generation_list.append(list(pop))
                g+=1
            # print(generation_list)
            return generation_list