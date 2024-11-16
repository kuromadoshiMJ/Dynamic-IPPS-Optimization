from breakdown_individual import Chromosome
from network_diagram import get_process_plan_matrix
from transform_data import get_alt_machines, get_machine_process_time, get_machine_energy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random

class NSGAIIBreakdown:
    def __init__(self, num_jobs, num_machines, len_plan_max, num_operations_arr, num_process_plan_arr, len_process_plan_arr, process_plan_matrix, process_time_matrix, energy_matrix, alt_machines_dict, population_size, generations, mutation_rate_1, mutation_rate_2, crossover_rate, bd_start = None, bd_end = None, bd_machine = None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.len_plan_max = len_plan_max
        self.num_process_plan_arr = num_process_plan_arr
        self.len_process_plan_arr = len_process_plan_arr
        self.num_operations_arr = num_operations_arr
        self.process_plan_matrix = process_plan_matrix
        self.process_time_matrix = process_time_matrix
        self.energy_matrix = energy_matrix
        self.alt_machines_dict = alt_machines_dict
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_1 = mutation_rate_1
        self.mutation_rate_2 = mutation_rate_2
        self.crossover_rate = crossover_rate
        self.bd_start = bd_start
        self.bd_end = bd_end
        self.bd_machine = bd_machine

        self.population = [self.random_individual() for _ in range(population_size)]

    def random_individual(self):
        chromosome = Chromosome(self.num_jobs, self.len_plan_max, self.num_operations_arr, self.bd_start, self.bd_end, self.bd_machine) 
        chromosome.encode_chromosome(self.num_jobs, self.num_process_plan_arr, self.len_process_plan_arr, self.num_operations_arr,self.len_plan_max, self.alt_machines_dict)
        chromosome.decode_chromosome(self.num_jobs, self.num_machines, self.len_plan_max, self.process_plan_matrix, self.process_time_matrix, self.energy_matrix)
        return chromosome

    def crossover(self, parent_1, parent_2, num_jobs, len_plan_max, num_operations_arr):
        # print("Parent 1")
        # print(parent_1.process_plan_gene)
        # print(parent_1.schedule_plan_gene)
        # print(parent_1.operation_machine_gene)
        # print('\n')
        # print("Parent 2")
        # print(parent_2.process_plan_gene)
        # print(parent_2.schedule_plan_gene)
        # print(parent_2.operation_machine_gene)
        # print('\n\n')
        child_1 = Chromosome(num_jobs, len_plan_max, num_operations_arr, self.bd_start, self.bd_end, self.bd_machine)
        child_2 = Chromosome(num_jobs, len_plan_max, num_operations_arr, self.bd_start, self.bd_end, self.bd_machine)
        
        # initialize schedule plan to -1
        child_1.schedule_plan_gene = child_1.schedule_plan_gene-1
        child_2.schedule_plan_gene = child_2.schedule_plan_gene-1

        # set process plan
        common_idx = []
        for i in range(num_jobs):
            if parent_1.process_plan_gene[i] == parent_2.process_plan_gene[i]:
                common_idx.append(i)
        
        for i in range(num_jobs):
            if( i in common_idx):
                child_1.process_plan_gene[i] = parent_1.process_plan_gene[i]
                child_2.process_plan_gene[i] = parent_2.process_plan_gene[i]
            else:
                child_1.process_plan_gene[i] = parent_2.process_plan_gene[i]
                child_2.process_plan_gene[i] = parent_1.process_plan_gene[i]

        empty_space_1 = num_jobs * len_plan_max
        empty_space_2 = num_jobs * len_plan_max

        for i in range(num_jobs * len_plan_max):
            
            if (parent_1.schedule_plan_gene[i]-1) in common_idx or parent_1.schedule_plan_gene[i] == 0:
                child_1.schedule_plan_gene[i] = parent_1.schedule_plan_gene[i]
                empty_space_1 -= 1
            if (parent_2.schedule_plan_gene[i]-1) in common_idx or parent_2.schedule_plan_gene[i] == 0:
                child_2.schedule_plan_gene[i] = parent_2.schedule_plan_gene[i]
                empty_space_2 -= 1
        
        # print(empty_space_1, empty_space_2)
        if empty_space_1 >= empty_space_2:
            unassigned_idx = np.where(child_1.schedule_plan_gene == -1)[0]
            # print(unassigned_idx)
            selected_idx = np.random.choice(unassigned_idx, size = (empty_space_1 - empty_space_2), replace = False)
            # print(selected_idx)
            child_1.schedule_plan_gene[selected_idx] == 0
        else:
            unassigned_idx = np.where(child_2.schedule_plan_gene == -1)[0]

            selected_idx = np.random.choice(unassigned_idx, size = (empty_space_2 - empty_space_1), replace = False)

            child_2.schedule_plan_gene[selected_idx] == 0

            
        i = 0
        j = 0

        while i< (num_jobs * len_plan_max) and j < (num_jobs * len_plan_max):
            if (parent_2.schedule_plan_gene[j]-1 in common_idx) or parent_2.schedule_plan_gene[j] == 0:
                j+=1
            elif(child_1.schedule_plan_gene[i] != -1):
                i+=1
            else:
                child_1.schedule_plan_gene[i] = parent_2.schedule_plan_gene[j]
                i+=1
                j+=1

        i = 0
        j = 0

        while i< (num_jobs * len_plan_max) and j < (num_jobs * len_plan_max):
            if(parent_1.schedule_plan_gene[j]-1 in common_idx) or parent_1.schedule_plan_gene[j] == 0:
                j+=1
            elif(child_2.schedule_plan_gene[i] != -1):
                i+=1
            else:
                child_2.schedule_plan_gene[i] = parent_1.schedule_plan_gene[j]
                i+=1
                j+=1
        

        for i in range(len(parent_1.operation_machine_gene)):
            array1 = parent_1.operation_machine_gene[i]
            array2 = parent_2.operation_machine_gene[i]

            array3 = parent_1.operation_tool_gene[i]
            array4 = parent_2.operation_tool_gene[i]

            cross_idx = np.random.choice(len(array1))
            
            newarr1 = np.concatenate((array1[:cross_idx+1], array2[cross_idx+1:]))
            newarr2 = np.concatenate((array2[:cross_idx+1], array1[cross_idx+1:]))

            newarr3 = np.concatenate((array3[:cross_idx+1], array4[cross_idx+1:]))
            newarr4 = np.concatenate((array4[:cross_idx+1], array3[cross_idx+1:]))
            

            child_1.operation_machine_gene[i] = newarr1
            child_2.operation_machine_gene[i] = newarr2

            child_1.operation_tool_gene[i] = newarr3
            child_2.operation_tool_gene[i] = newarr4
        # print("child_1 =\n")
        # print(child_1.process_plan_gene)
        # print(child_1.schedule_plan_gene)
        # print(child_1.operation_machine_gene)
        # print(child_1.operation_tool_gene)
        # print('\n child2=\n')
        # print(child_2.process_plan_gene)
        # print(child_2.schedule_plan_gene)
        # print(child_2.operation_machine_gene)
        # print(child_2.operation_tool_gene)

        child_1.decode_chromosome(self.num_jobs, self.num_machines, self.len_plan_max, self.process_plan_matrix, self.process_time_matrix, self.energy_matrix)
        child_2.decode_chromosome(self.num_jobs, self.num_machines, self.len_plan_max, self.process_plan_matrix, self.process_time_matrix, self.energy_matrix)
        return child_1, child_2
    
    def schedule_mutation_1(self, parent):
        child = copy.deepcopy(parent)

        idx1, idx2 = np.random.choice(len(child.schedule_plan_gene), size=2, replace=False)

        # Swap the elements at idx1 and idx2
        child.schedule_plan_gene[idx1], child.schedule_plan_gene[idx2] = child.schedule_plan_gene[idx2], child.schedule_plan_gene[idx1]
        child.decode_chromosome(self.num_jobs, self.num_machines, self.len_plan_max, self.process_plan_matrix, self.process_time_matrix, self.energy_matrix)
        return child
    
    def schedule_mutation_2(self, parent, num_process_plan_arr):
        child = copy.deepcopy(parent)

        idx = np.random.choice(len(child.process_plan_gene))
        mutated_value = np.random.randint(1, num_process_plan_arr[idx] + 1)

        child.process_plan_gene[idx] = mutated_value
        child.decode_chromosome(self.num_jobs, self.num_machines, self.len_plan_max, self.process_plan_matrix, self.process_time_matrix, self.energy_matrix)
        return child
    
    def operation_mutation(self, parent, alt_machine_dict):
        child = copy.deepcopy(parent)
        job_idx = np.random.choice(len(child.operation_machine_gene))

        op_idx = np.random.choice(len(child.operation_machine_gene[job_idx]))

        child.operation_machine_gene[job_idx][op_idx] = np.random.choice(alt_machine_dict[job_idx+1][op_idx+1])
        child.decode_chromosome(self.num_jobs, self.num_machines, self.len_plan_max, self.process_plan_matrix, self.process_time_matrix, self.energy_matrix)
        return child
    
    def run(self):
        for generation in range(self.generations):
            print("Generation =", generation)
            print(generation)
            for pop in nsga2.population:
                print("Start Chromosome \n")
                print(pop.process_plan_gene)
                print(pop.schedule_plan_gene)
                print(pop.operation_machine_gene)
                print(pop.objectives, pop.rank, pop.crowding_distance)
                print("End Chromsome \n")
            # crossover
            self.offsprings = []

            for _ in range(int(self.population_size/2)):
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = self.crossover(parent_1= parent1, parent_2=parent2, num_jobs=num_jobs, len_plan_max=len_plan_max, num_operations_arr= num_operations_arr)
                self.offsprings.append(child1)
                self.offsprings.append(child2)

            # mutation
            for ind in self.offsprings:
                # do mutation_1
                if (random.random() < self.mutation_rate_1):
                    ind = self.operation_mutation(ind, self.alt_machines_dict)

                else:
                    if(random.random() < self.mutation_rate_2):
                        if(random.random() <0.5):
                            ind = self.schedule_mutation_1(ind)
                        else:
                            ind = self.schedule_mutation_2(ind, self.num_process_plan_arr)

            # combined population
            combined_population = self.population + self.offsprings
            
            # non-dominated sorting
            self.non_dominated_sorting(combined_population)
            # for pop in nsga2.population:
                # print(pop.objectives, pop.rank, pop.crowding_distance)
            self.population = self.select_next_generation(combined_population)
            
            
        return self.get_pareto_front(self.population)
        
    def non_dominated_sorting(self, combined_population):
        fronts = [[]]

        for i in range(len(combined_population)):
            p = combined_population[i]
            p.domination_count = 0
            p.dominated_solutions = []

            for j in range(len(combined_population)):
                q = combined_population[j]
                if i != j:
                    if np.all(p.objectives <= q.objectives) and np.any(p.objectives < q.objectives):
                        p.dominated_solutions.append(q)
                    elif np.all(q.objectives <= p.objectives) and np.any(q.objectives < p.objectives):
                        p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # print(fronts)
        self.calculate_crowding_distance(fronts)

    def calculate_crowding_distance(self, fronts):
        for front in fronts:
            if len(front) == 0:
                continue

            for ind in front:
                ind.crowding_distance = 0

            objectives = np.array([ind.objectives for ind in front])
            for m in range(objectives.shape[1]):
                sorted_indices = np.argsort(objectives[:, m])
                front[sorted_indices[0]].crowding_distance = float('inf')
                front[sorted_indices[-1]].crowding_distance = float('inf')

                for i in range(1, len(front) - 1):
                    front[sorted_indices[i]].crowding_distance += (objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m])


    def select_next_generation(self, combined_population):
        # Sort population by rank and then by crowding distance
        combined_population.sort(key=lambda x: (x.rank, -x.crowding_distance))
        return combined_population[:self.population_size]

    def get_pareto_front(self, population):
        return np.array([ind.objectives for ind in population if ind.rank == 0])

def load_excel_data(filepath):
    sheets = pd.read_excel(filepath, sheet_name=['Job1', 'Job2', 'Job3'])
    return sheets

def plot_pareto_front(pareto_front):
    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='b', marker='o', label='Pareto Front')
    plt.title('Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__" :

    excel_filepath = 'D:\MTP\GA\DATASET_NEW.xlsx'
    sheets_data = load_excel_data(excel_filepath)
    num_jobs = 3
    len_plan_max = 8
    num_machines = 17
    
    num_process_plan_arr = np.array([6, 8, 4])
    len_process_plan_arr = np.array([5, 6, 8])
    num_operations_arr = np.array([11, 14, 17])

    process_plan_matrix = get_process_plan_matrix()
    process_time_matrix = get_machine_process_time(sheets_data, num_jobs)
    energy_matrix  = get_machine_energy(sheets_data, num_jobs)

    alt_machines_dict = get_alt_machines(sheets_data, num_jobs)

    # print(process_time_matrix)
    
    
    # my_chromosome_1 = Chromosome(num_jobs, len_plan_max, num_operations_arr)
    # my_chromosome_1.process_plan_gene = np.array([2, 2, 2])
    # my_chromosome_1.schedule_plan_gene = np.array([1, 1, 1, 3, 3, 1, 3, 2, 3, 3, 0, 0, 0, 3, 2, 1, 2, 2, 2, 3, 0, 0, 3, 2])
    # my_chromosome_1.operation_machine_gene = [np.array([ 3,  2,  3,  1,  7,  6,  7, 10, 11,  9,  8]), np.array([ 3,  1,  1,  4,  4,  6,  6,  5,  9,  9, 11, 12,  7,  9]), np.array([ 2,  2,  3,  1,  2,  5,  8,  9, 10,  8,  9, 14, 13,  9,  7, 12, 11])]
    # my_chromosome_1.operation_tool_gene = [np.array([2, 2, 0, 1, 0, 2, 0, 2, 1, 0, 2]), np.array([0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 1, 1, 0, 1]), np.array([2, 1, 0, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 1, 1, 1, 1])]


    my_chromosome_2 = Chromosome(num_jobs, len_plan_max, num_operations_arr, 10, 50, 4)
    my_chromosome_2 = Chromosome(num_jobs, len_plan_max, num_operations_arr)
    my_chromosome_2.process_plan_gene = np.array([4, 3, 3])
    my_chromosome_2.schedule_plan_gene = np.array([0, 1, 2, 3, 1, 3, 2, 3, 0, 2, 0, 3, 2, 3, 0, 1, 3, 3, 1, 2, 1, 2, 0, 3])
    my_chromosome_2.operation_machine_gene = [np.array([ 3,  1,  1,  2,  7,  7,  6,  7, 12,  7,  8]), np.array([ 2,  3,  1,  4,  1,  8,  9,  7,  7,  8, 12, 12,  6,  5]), np.array([ 1,  2,  4,  1,  2,  7,  8,  6, 10,  8,  8, 13, 14,  9,  6, 12, 11])]
    my_chromosome_2.operation_tool_gene = [np.array([0, 2, 1, 0, 1, 1, 0, 0, 2, 1, 1]), np.array([1, 0, 0, 0, 2, 2, 2, 0, 2, 0, 1, 2, 0, 2]), np.array([1, 0, 0, 2, 2, 1, 0, 1, 1, 0, 0, 2, 0, 0, 2, 2, 1])]
    
    my_chromosome_2.decode_chromosome(num_jobs, num_machines, len_plan_max, process_plan_matrix, process_time_matrix, energy_matrix)
    my_chromosome_2.plot_gantt_chart()
    # nsga2 = NSGAII(num_jobs, num_machines, len_plan_max, num_operations_arr, num_process_plan_arr, len_process_plan_arr, process_plan_matrix, process_time_matrix, energy_matrix, alt_machines_dict, population_size = 2, generations=10, mutation_rate_1=0.3, mutation_rate_2=0.3, crossover_rate=0.8)
    

    # for i in range(2):
    #     print(nsga2.population[i].process_plan_gene)
    #     print(nsga2.population[i].schedule_plan_gene)
    #     print(nsga2.population[i].operation_machine_gene)
    #     print(nsga2.population[i].operation_tool_gene)

    # nsga2.crossover(nsga2.population[0], nsga2.population[1], num_jobs, len_plan_max, num_operations_arr)
    # my_chromosome = Chromosome(num_jobs, len_plan_max, num_operations_arr)
    # my_chromosome.encode_chromosome(num_jobs, num_process_plan_arr, len_process_plan_arr, num_operations_arr, len_plan_max, alt_machines_dict)
    # my_chromosome.decode_chromosome(num_jobs, num_machines, len_plan_max, process_plan_matrix, process_time_matrix, energy_matrix)
    # my_chromosome.plot_gantt_chart()
    # nsga2 = NSGAIIBreakdown(num_jobs, num_machines, len_plan_max, num_operations_arr, num_process_plan_arr, len_process_plan_arr, process_plan_matrix, process_time_matrix, energy_matrix, alt_machines_dict, population_size = 50, generations = 20, mutation_rate_1=0.3, mutation_rate_2=0.3, crossover_rate=0.8)
    
    # print(len(nsga2.population))
    # print("My Pop .print")
    # for pop in nsga2.population:
    #     print("Start Chromosome \n")
    #     print(pop.process_plan_gene)
    #     print(pop.schedule_plan_gene)
    #     print(pop.operation_machine_gene)
    #     print("End Chromsome \n")
    #     print(pop.objectives, pop.rank, pop.crowding_distance)
    pareto_front = nsga2.run()

    print("My Pop .print")
    for pop in nsga2.population:
        print(pop.objectives, pop.rank, pop.crowding_distance)

    nsga2.population[0].plot_gantt_chart()
    # # print(len(nsga2.population))
    # for pop in nsga2.population:
    #     print("Start Chromosome \n")
    #     print(pop.process_plan_gene)
    #     print(pop.schedule_plan_gene)
    #     print(pop.operation_machine_gene)
    #     print("End Chromsome \n")
    #     print(pop.objectives, pop.rank, pop.crowding_distance)
    # print(pareto_front)
    # print(len(pareto_front))
    # plot_pareto_front(pareto_front)
    
    