import numpy as np
import pandas as pd
import re
import copy
from transform_data import get_alt_machines, get_machine_process_time, get_machine_energy
from network_diagram import get_process_plan_matrix

from initial_population import Chromosome
import matplotlib.pyplot as plt
def load_excel_data(filepath):
    
    sheets = pd.read_excel(filepath, sheet_name=['Job1', 'Job2', 'Job3'])
    return sheets


def decode_chromosome(chromosome, num_jobs, len_plan_max, process_plan_matrix, process_time_matrix):
    
    schedule_plan_arr = chromosome.schedule_plan_gene
    process_plan_arr = chromosome.process_plan_gene
    operation_machine_chromosome = chromosome.operation_machine_gene

    print("operationMC", operation_machine_chromosome)
    freq_dict = {}
    machine_engage_time_dict = {}
    machine_last_completion_time_arr = np.zeros(17)
    allowable_start_time = np.zeros((num_jobs, len_plan_max))
    start_time = np.zeros((num_jobs, len_plan_max))
    completion_time = np.zeros((num_jobs, len_plan_max))
    # print(schedule_plan_arr)
    for i in schedule_plan_arr:
        if i == 0:
            continue
        if freq_dict.get(i) is None:
            freq_dict[i] = 1
            allowable_start_time[i-1][freq_dict[i]-1] = 0
                
        else:
            freq_dict[i] = freq_dict[i]+1
            allowable_start_time[i-1][freq_dict[i]-1] = completion_time[i-1][(freq_dict[i]-1)-1]
        
        match = re.search(r'O(\d+)', process_plan_matrix[i-1][process_plan_arr[i-1]-1][freq_dict[i]-1])
        if match:
            operation_id = int(match.group(1))
        current_machine_id = operation_machine_chromosome[i-1][operation_id-1]
        process_time = process_time_matrix[i][operation_id][current_machine_id][0]
        if machine_engage_time_dict.get(current_machine_id) is None:
                machine_engage_time_dict[current_machine_id] = []
        if(len(machine_engage_time_dict[current_machine_id]) == 0):
            start_time[i-1][freq_dict[i]-1] = allowable_start_time[i-1][freq_dict[i]-1]
            
        else:
            idle_s = 0
            flag = 0
            for j in range(len(machine_engage_time_dict[current_machine_id])):
                idle_e = machine_engage_time_dict[current_machine_id][j][0]
                if(max(allowable_start_time[i-1][freq_dict[i]-1], idle_s) + process_time <= idle_e):
                    start_time[i-1][freq_dict[i]-1] = max(allowable_start_time[i-1][freq_dict[i]-1], idle_s)
                    flag = 1
                    break
                idle_s = machine_engage_time_dict[current_machine_id][j][1]
                
            if(flag == 0):
                start_time[i-1][freq_dict[i]-1] = max(allowable_start_time[i-1][freq_dict[i]-1], machine_last_completion_time_arr[current_machine_id-1])

        completion_time[i-1][freq_dict[i]-1] = start_time[i-1][freq_dict[i] - 1] + process_time
        machine_engage_time_dict[current_machine_id].append([start_time[i-1][freq_dict[i]-1], completion_time[i-1][freq_dict[i]-1], i, operation_id]) 
        machine_engage_time_dict[current_machine_id].sort()
        machine_last_completion_time_arr[current_machine_id-1] = max(machine_last_completion_time_arr[current_machine_id-1],  completion_time[i-1][freq_dict[i]-1])

    return np.max(machine_last_completion_time_arr)
    print(machine_engage_time_dict)


    data =[]

    colo_mapping = {
        1 : 'blue',
        2 : 'green',
        3 : 'orange'
    }

    for machine_id, operations in machine_engage_time_dict.items():
        for operation in operations:
            # print(operation)
            start, end, job_id, op_id = operation
            color = colo_mapping.get(job_id, 'gray')
            data.append({"Machine ID": machine_id, "Start": start, "End": end, 'j_id': job_id, 'color': color})

    df = pd.DataFrame(data)

    # Create a Gantt chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each machine's operations and create bars for each operation
    for index, row in df.iterrows():
        ax.barh(row['Machine ID'], row['End'] - row['Start'], left=row['Start'], align='center', color = row['color'])

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine ID')
    ax.set_title('Machine Operation Gantt Chart')

    # Show grid
    ax.grid(True)
    plt.show()
def crossover(parent_1, parent_2, num_jobs, len_plan_max, num_operations_arr):
    child_1 = Chromosome(num_jobs, len_plan_max, num_operations_arr)
    child_2 = Chromosome(num_jobs, len_plan_max, num_operations_arr)
    child_1.schedule_plan_gene = child_1.schedule_plan_gene-1
    child_2.schedule_plan_gene = child_2.schedule_plan_gene-1

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
    
    if empty_space_1 >= empty_space_2:
        unassigned_idx = np.where(child_1.schedule_plan_gene == -1)[0]

        selected_idx = np.random.choice(unassigned_idx, size = (empty_space_1 - empty_space_2), replace = False)

        child_1.schedule_plan_gene[selected_idx] == 0
    else:
        unassigned_idx = np.where(child_2.schedule_plan_gene == -1)[0]

        selected_idx = np.random.choice(unassigned_idx, size = (empty_space_2 - empty_space_1), replace = False)

        child_2.schedule_plan_gene[selected_idx] == 0

        
    i = 0
    j = 0

    while i< (num_jobs * len_plan_max) and j < (num_jobs * len_plan_max):
        if(parent_2.schedule_plan_gene[j] not in common_idx):
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
        if(parent_1.schedule_plan_gene[j] not in common_idx):
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

        cross_idx = np.random.choice(len(array1))
        
        newarr1 = np.concatenate((array1[:cross_idx+1], array2[cross_idx+1:]))
        newarr2 = np.concatenate((array2[:cross_idx+1], array1[cross_idx+1:]))
        
        child_1.operation_machine_gene[i] = newarr1
        child_2.operation_machine_gene[i] = newarr2
    return child_1, child_2
        
def schedule_mutation_1(parent):
    child = copy.deepcopy(parent)

    idx1, idx2 = np.random.choice(len(child.schedule_plan_gene), size=2, replace=False)

    # Swap the elements at idx1 and idx2
    child.schedule_plan_gene[idx1], child.schedule_plan_gene[idx2] = child.schedule_plan_gene[idx2], child.schedule_plan_gene[idx1]

    return child
def schedule_mutation_2(parent, num_process_plan_arr):
    child = copy.deepcopy(parent)

    idx = np.random.choice(len(child.process_plan_gene))
    mutated_value = np.random.randint(1, num_process_plan_arr[idx] + 1)

    child.process_plan_gene[idx] = mutated_value

    return child

def operation_mutation(parent, alt_machine_dict):
    child = copy.deepcopy(parent)
    job_idx = np.random.choice(len(child.operation_machine_gene))

    op_idx = np.random.choice(len(child.operation_machine_gene[job_idx]))

    child.operation_machine_gene[job_idx][op_idx] = np.random.choice(alt_machine_dict[job_idx+1][op_idx+1])

    return child
def main():

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
    energy_time_matrix  = get_machine_energy(sheets_data, num_jobs)
    print(energy_time_matrix)
    return 1
    alt_machines_dict = get_alt_machines(sheets_data, num_jobs)

    for i in process_plan_matrix:
        print(i)
    # print(alt_machines_dict)
    for key in  alt_machines_dict:
        print(key, alt_machines_dict[key])
    my_first_chromosome = Chromosome(num_jobs, len_plan_max, num_operations_arr)
    # my_first_chromosome.fill_chromosome(num_jobs, num_process_plan_arr, len_process_plan_arr, num_operations_arr, len_plan_max, alt_machines_dict)

    my_first_chromosome.process_plan_gene = np.array([2, 6, 3])
    my_first_chromosome.schedule_plan_gene = np.array([3, 2, 1, 1, 0, 3, 2, 2, 1, 2, 3, 3, 2, 3, 2, 0, 0, 1, 2, 3, 0, 1, 0, 3])
    my_first_chromosome.operation_machine_gene = [np.array([ 3,  2,  1,  2,  7,  6,  7, 5, 11,  9,  9]), np.array([ 2,  1,  1,  4,  1,  7,  9,  10,  8,  9, 11, 12,  10, 5]), np.array([ 2, 3, 3, 1, 2, 5, 7, 9, 5, 8, 9, 13, 13, 9,  9, 12, 11])]

    print(my_first_chromosome.process_plan_gene)
    print(my_first_chromosome.schedule_plan_gene)
    print(my_first_chromosome.operation_machine_gene, "\n")

    my_second_chromosome = Chromosome(num_jobs, len_plan_max, num_operations_arr)

    my_second_chromosome.process_plan_gene = np.array([6, 1, 3])
    my_second_chromosome.schedule_plan_gene = np.array([1, 3, 3, 2, 1, 3, 2, 2, 3, 0, 0, 0, 1, 1, 2, 3, 2, 0, 3, 2, 0, 1, 3, 3])
    my_second_chromosome.operation_machine_gene = [np.array([ 2,  1,  4,  2,  7,  8,  7, 10, 11,  9,  7]), np.array([ 3,  3,  3,  2,  1,  8,  6,  7,  9,  8, 12, 12,  7,  7]), np.array([ 3,  1,  4,  2,  3,  7,  8,  7,  5,  9,  9, 13, 14,  9,  6, 11, 12])]
    
    # print(my_second_chromosome.process_plan_gene)
    # print(my_second_chromosome.schedule_plan_gene)
    print(my_second_chromosome.operation_machine_gene)

    child1, child2 = crossover(my_first_chromosome, my_second_chromosome, num_jobs, len_plan_max, num_operations_arr)
    # print(child1.process_plan_gene)
    # print(child1.schedule_plan_gene)
    # print(child1.operation_machine_gene)

    # print(child2.process_plan_gene)
    # print(child2.schedule_plan_gene)
    # print(child2.operation_machine_gene)

    # child_m = schedule_mutation_1(child1)
    # child_m2 = schedule_mutation_2(child2, num_process_plan_arr)
    child_m3 = operation_mutation(my_first_chromosome, alt_machines_dict)
    # print(child_m.process_plan_gene)
    # print(child_m.schedule_plan_gene)

    # print(child_m2.process_plan_gene)
    # print(child_m2.schedule_plan_gene)

    print(child_m3.process_plan_gene)
    print(child_m3.schedule_plan_gene)
    print(child_m3.operation_machine_gene)

    decode_chromosome_2(my_second_chromosome, num_jobs, len_plan_max, process_plan_matrix, process_time_matrix)
main()