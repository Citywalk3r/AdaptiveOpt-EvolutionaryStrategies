import random
import numpy as np
from queue import Queue

class Evolutionary_Strategies:

    def __init__(self):
        pass

    def generate_init_pop(self, seed, initial_sigma):
        """
        Generates the initial population given population size m.
        """
        rng = np.random.default_rng(seed)
        init_generation = [[*np.ndarray.tolist(rng.uniform(-32,32,2)), *[initial_sigma, initial_sigma]] for _ in range(self.m)]
        return init_generation

    def calculate1fifth(self):
        """
        Calculates the success rate of children over parent
        """
        truth_table = np.asarray(list(self.q.queue))
        successful_children = np.count_nonzero(truth_table)
        success_rate = successful_children/truth_table.size
        return success_rate

    def mutation_sigma_recombination(self, parent, pop):
            """Implements the mutation operator.
            Parameters:
                parent : 2D array [x,y]
            Returns:
                children : list of children arrays [[x1,y1], [x2,y2] ... [x l//m,y l//m]]

            """
            children = []
            parent_score = self.eval_func(parent)

            for _ in range(self.l//self.m):

                # Randomly select 4 parents from the population
                parents_for_recomb = random.sample(pop, 4)

                # Get the sigma_x from 2 parents
                parent_1_sigma_x = parents_for_recomb[0][2]
                parent_2_sigma_x = parents_for_recomb[1][2]

                # Get the sigma_y from 2 parents
                parent_3_sigma_y = parents_for_recomb[2][3]
                parent_4_sigma_y = parents_for_recomb[3][3]

                

                child = parent.copy()
       
                if(self.mutations_performed%self.n == 0 and self.q.full()):
                    success = self.calculate1fifth()
                    if success > 0.2:
                        self.std_dev_multiplier = self.std_dev_multiplier/0.85
                    elif success < 0.2:
                        self.std_dev_multiplier = self.std_dev_multiplier*0.85
                    self.std_dev_multiplier_list.append(self.std_dev_multiplier)

                parent_1_sigma_x = parent_1_sigma_x*self.std_dev_multiplier
                parent_2_sigma_x = parent_2_sigma_x*self.std_dev_multiplier
                parent_3_sigma_y = parent_3_sigma_y*self.std_dev_multiplier
                parent_4_sigma_y = parent_4_sigma_y*self.std_dev_multiplier
 

                child[2] = (parent_1_sigma_x + parent_2_sigma_x)/2
                child[3] = (parent_3_sigma_y + parent_4_sigma_y)/2

                modifier_x = np.random.normal(loc=0.0, scale=child[2])
                modifier_y = np.random.normal(loc=0.0, scale=child[3])

                if child[0] + modifier_x <=32 and child[0] + modifier_x >=-32: # Check if out of boundaries
                    child[0] += modifier_x
                    

                if child[1] + modifier_y <=32 and child[1] + modifier_y >=-32: # Check if out of boundaries
                    child[1] += modifier_y
                
                if self.q.full():
                    # Remove element from the queue
                    self.q.get()

                # Insert the last mutation to the queue
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
            self.sigma = initial_sigma
            self.mutations_performed = 0
            self.std_dev_multiplier_list=[]
            self.l = l
            self.m = m
            self.std_dev_multiplier = 1

            # Initialize a queue to keep last 10*n mutations
            self.q = Queue(maxsize = 10*n)

            # final populations for each generation (used for visualization)
            generation_list = []
            g = 0
            initial_population = self.generate_init_pop(seed, initial_sigma)

            # endregion
            pop = initial_population

            while (g<g_max):
    
                if(strategy == "k,l"):
                    pop = []
                for _ in range (m):
                    parent = random.choice(pop)
                    # for parent in pop:
                    children = self.mutation_sigma_recombination(parent, pop)
                    pop.extend(children)
                pop = self.select_m_best(pop)
                generation_list.append(list(pop))
                g+=1
            return generation_list, self.std_dev_multiplier_list