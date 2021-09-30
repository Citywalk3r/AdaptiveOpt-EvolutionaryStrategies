import numpy as np
from evolutionary_strategies import Evolutionary_Strategies
from evolutionary_strategies_recomb import Evolutionary_Strategies as EV_Recomb_sigma
from evolutionary_strategies_recomb_decision_vars import Evolutionary_Strategies as EV_Recomb_vars
import matplotlib.pyplot as plt
import math
import random
import csv
from queue import Queue
import pandas as pd

class ACKLEY:

    def __init__(self, is_debug):
        self.is_debug = is_debug
    
    
    def tournament_selection(self, population,  eval_f):
        """
        To find the first parent, selects 3 parents from the population and returns the best.
        To find the second parent, selects k parents from the population and returns the best.
        """
        k=40
        parent1 = random.sample(population, 2)
        parent1 = sorted(parent1, key=lambda parent: eval_f(parent))

        parent2 = random.sample(population, k)
        parent2 = sorted(parent2, key=lambda parent: eval_f(parent))
        return parent1[0], parent2[0]
    

    def single_point_crossover(self, p1, p2):
        """
        Implements the crossover function.
        Single point crossover with repair for infeasible solutions.
        """

        modifier = np.random.normal(loc=0.0, scale=0.001)

        dim = np.random.randint(2)

        M = np.random.randint(1,1001)

        delta1 = p1[dim]/M
        delta2 = p2[dim]/M

        child1 = p1.copy()
        child2 = p2.copy()

        if (p1[dim] + delta2 - delta1 <= 32 and p1[dim] + delta2 - delta1 >= -32 ):
            child1[dim] = p1[dim] + delta2 - delta1

        if (p2[dim] - delta2 + delta1 <= 32 and p2[dim] - delta2 + delta1 >= -32 ):
            child2[dim] = p2[dim] - delta2 + delta1

        if self.is_debug:
            print("Parent 1: ", p1)
            print("Parent 2: ", p2)

            print("modifier: ", modifier)

            print("Child 1 : ", child1)
            print("Child 2 : ", child2)
        
        return child1, child2


    



    def eval_func(self, individual):
        """
        Evaluates the current state by
        calculating the function result.
        """
        # print(individual)

        x = individual[0]
        y = individual[1]
        f = -20 * math.exp(-0.2 * math.sqrt(0.5*(x**2+y**2))) - math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))) + 20 + math.e
        return f
    
    def solve_ackley(self):
        """
        Calls the GA for the Ackley function problem and plots the results.
        """
      
        fig, axs = plt.subplots(5, 6)

        p_cross = 0.9
        p_mut_list = [0.9, 0.5, 0.1]
        p_size_list = [50, 100, 150]
        g_max_list = [300, 400, 500]

        colors=['xkcd:light cyan','xkcd:light seafoam','xkcd:very light blue','xkcd:eggshell','xkcd:very light purple',
        'xkcd:light light blue','xkcd:lightblue','xkcd:buff','xkcd:light mint','xkcd:light periwinkle',]

        with open('../results_ackley2.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["gen_max","pop_size","mut_prob","0","1","2","3","4","5","6","7","8","9"])

            for g_max in g_max_list:
                for p_idx, p_size in enumerate(p_size_list):
                    for mut_p in p_mut_list:
                        write_string = [g_max, p_size, mut_p]
                        for i in range(10):
                            generation_list = es(ipop_f=self.generate_init_pop,
                                                p_cross=p_cross,
                                                p_mut=mut_p, p_size=p_size,
                                                g_max=g_max,
                                                eval_f= self.eval_func,
                                                seed=i,
                                                selection_f=self.tournament_selection,
                                                crossover_f=self.single_point_crossover,
                                                mutation_f=self.mutate_degrading)


                            generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
                            gen_eval_list_np = np.array(generation_eval_list)
                    
                            # list of mins of each generation
                            gen_mins = np.min(gen_eval_list_np, 1)
            
                            # best solution
                            best = np.min(gen_mins)
                            write_string.append(best)


                        writer.writerow(write_string)

        #             axs.flat[i*3 + p_idx].plot(range(g_max), gen_mins, label="p_mut: "+str(mut_p))

        #         best_per_solution = np.min(best_per_pmut)

        #         plot_idx = i*3 + p_idx
        #         axs.flat[plot_idx].set_title('s={:},p_size={:},g_max={:}'.format(i,p_size,g_max))
        #         axs.flat[plot_idx].set(xlabel="generations", ylabel="best individual score")
        #         axs.flat[plot_idx].hlines(y=best_per_solution, xmin=0, xmax=g_max, linewidth=2, color='r', label="best: "+str(best_per_solution))
        #         axs.flat[plot_idx].legend()
        #         axs.flat[plot_idx].label_outer()
        #         axs.flat[plot_idx].set_facecolor(colors[i])
        
        # plt.show()


        # p_cross = 0.9
        # p_mut = 0.9
        # p_size = 200
        # g_max = 300

        # generation_list = ga(ipop_f=self.generate_init_pop,
        #                                 p_cross=p_cross,
        #                                 p_mut=p_mut, p_size=p_size,
        #                                 g_max=g_max,
        #                                 eval_f= self.eval_func,
        #                                 seed=0,
        #                                 selection_f=self.tournament_selection,
        #                                 crossover_f=self.single_point_crossover,
        #                                 mutation_f=self.mutate_degrading)


        # generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
        # gen_eval_list_np = np.array(generation_eval_list)
    
        # # list of mins of each generation
        # gen_mins = np.min(gen_eval_list_np, 1)
        # # print(generation_eval_list)
        # # print(gen_mins)

        # # best solution
        # best = np.min(gen_mins)
        # print("Best solution: ", best)

        
        # plt.title('p_size={:},g_max={:}'.format(p_size,g_max))
        # plt.axes(xlabel="generations", ylabel="best individual score")
        # plt.plot(range(g_max), gen_mins)
        # plt.hlines(y=best, xmin=0, xmax=g_max, linewidth=2, color='r', label="best: "+str(best))
        # plt.legend()
        # plt.show()

    def do_one_trial(self, g_max = 15):

        Ev = Evolutionary_Strategies()
        Ev_recomb = EV_Recomb_vars()
        data = []

        headers = ['σ', 1821, 97, 1940, 1924, 1250, 776, 600, 430, 445, 336]
        seeds = [1821, 1453, 1940, 1924, 1250, 776, 600, 430, 445, 336]
        l_to_m_ratios = [(1000, 6000), (1400,9800), (2000,16000)]
        init_std_devs = [0.1]
        solutions_for_every_seed = []
        # for index, tuple in enumerate(l_to_m_ratios):
        for dev in init_std_devs:
            for seed in seeds:
                generation_list, sigma_list = Ev_recomb.es(m=2000, l=16000, g_max=g_max, eval_f= self.eval_func, seed=seed, initial_sigma=dev, 
                recombination_f=None, strategy="k+l", n=5)

                generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
                gen_eval_list_np = np.array(generation_eval_list)
                # print(generation_eval_list)

                # list of mins of each generation
                gen_mins = np.min(gen_eval_list_np, 1)

                # best solution
                best = np.min(gen_mins)
                solutions_for_every_seed.append(best)
                print("Best solution: ", best)

            tmp = [dev]
            tmp.extend(solutions_for_every_seed)
            data.append(tmp)
            solutions_for_every_seed = []

        print(data)
        df= pd.DataFrame(data=data,
                    columns= list(map(str, headers)))
        print(df)
        df.to_excel("../dec_var_recombination_10_seeds.xlsx")

        # plt.title('g_max={:}'.format(g_max))
        # plt.axes(xlabel="generations", ylabel="best individual score")
        # plt.plot(range(g_max), gen_mins)
        # plt.hlines(y=best, xmin=0, xmax=g_max, linewidth=2, color='r', label="best: "+str(best))
        # plt.legend()
        # plt.show()

        # # print(sigma_list)
        # plt.title('sigma')
        # plt.axes(xlabel="number of adjustments", ylabel="sigma value")
        # plt.plot(sigma_list)
        # plt.yscale('log')
        # plt.show()
        

if __name__ == "__main__":
    ACKLEY = ACKLEY(is_debug=False)
    # ACKLEY.solve_ackley()
    ACKLEY.do_one_trial()