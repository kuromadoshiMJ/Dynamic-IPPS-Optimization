from Precedence_relation import Precedence
from network_diagram import get_process_plan_matrix
import pandas as pd
import numpy as np
import re
from transform_data import get_alt_machines, get_machine_process_time
# Global dictionary to hold the sequence data
Sequence_dict = {}

# Function to load Excel data into separate dataframes for each sheet
def load_excel_data(filepath):
    
    sheets = pd.read_excel(filepath, sheet_name=['Job1', 'Job2', 'Job3'])
   
    return sheets

# Function to handle the Sequence
def load_sequence():
    global Sequence_dict  # Declare that we are modifying the global Sequence_dict

    # Creating the list sequence_list
    sequence_list = []
    sequence_list.append(Precedence("precedence_table.xlsx", "Precedence1"))
    sequence_list.append(Precedence("precedence_table.xlsx", "Precedence2"))
    sequence_list.append(Precedence("precedence_table.xlsx", "Precedence3"))

    # Convert sequence_list list to a dictionary with simple keys
    Sequence_dict = {
        "Job1": sequence_list[0],
        "Job2": sequence_list[1],
        "Job3": sequence_list[2]
    }

    # Print the dictionary
    print("Sequence Dictionary:")
    print(Sequence_dict)

num_jobs = 3
len_plan_max = 8
num_machines = 16

num_process_plan_arr = np.array([6, 8, 4])
len_process_plan_arr = np.array([5, 6, 8])
num_operations_arr = np.array([11, 14, 17])

operation_machine_dict_list = []
operation_machine_tool_dict_list = []
process_plan_matrix = get_process_plan_matrix()
def generate_scheduling_chromosome():
    
    process_plan_arr = (np.random.randint(0, num_process_plan_arr))+1

    schedule_plan_list = []

    for i in range(1, num_jobs+1):
        schedule_plan_list.extend([i] * len_process_plan_arr[i-1])
    
    num_zeros = (num_jobs * len_plan_max) - len(schedule_plan_list)
    schedule_plan_list.extend([0] * num_zeros)
    schedule_plan_arr = np.array(schedule_plan_list)

    schedule_plan_arr = np.random.permutation(schedule_plan_arr)
    
    operation_machine_chromosome = []

    for i in range(num_jobs):
        job_id = i+1
        machine_plan = generate_machine_chromosome(job_id)
        operation_machine_chromosome.append(machine_plan)

    chromosome = {
                "process_plan_arr":process_plan_arr, 
                "scheduling_plan_arr": schedule_plan_arr,
                "operation_machine_chromosome": operation_machine_chromosome
                }
    return chromosome

def generate_machine_chromosome(job_id):
    
    machine_plan_arr = np.zeros(num_operations_arr[job_id - 1], dtype=int)
    print(len(operation_machine_dict_list))
    operation_machine_dict = operation_machine_dict_list[job_id -1]
    for i in range(num_operations_arr[job_id -1]):
        machine_plan_arr[i] = np.random.choice(operation_machine_dict[i+1])

    return machine_plan_arr
    
# print(generate_scheduling_chromosome())
def generate_initial_population():
    population_size = 10    
    initial_population_list = []

    for i in range(population_size):
        initial_population_list.append(generate_scheduling_chromosome())
    
    return initial_population_list


# Main part of the script
def decode_chromosome(chromosome):
   

    schedule_plan_arr = chromosome["scheduling_plan_arr"]
    process_plan_arr = chromosome["process_plan_arr"]
    operation_machine_chromosome = chromosome["operation_machine_chromosome"]
    print("operationMC", operation_machine_chromosome)
    freq_dict = {}
    machine_engage_time_dict = {}
    machine_last_completion_time_arr = np.zeros(16)
    allowable_start_time = np.zeros((num_jobs, len_plan_max))
    start_time = np.zeros((num_jobs, len_plan_max))
    completion_time = np.zeros((num_jobs, len_plan_max))
    for i in schedule_plan_arr:
        if i == 0:
            continue
        if freq_dict.get(i) is None:
            freq_dict[i] = 1
            match = re.search(r'O(\d+)', process_plan_matrix[i-1][process_plan_arr[i-1]-1][freq_dict[i]-1])
            if match:
                operation_id = int(match.group(1))
            allowable_start_time[i-1][freq_dict[i]-1] = 0
            current_machine_id = operation_machine_chromosome[i-1][operation_id-1]
            process_time = operation_machine_tool_dict_list[i][operation_id][current_machine_id][0]
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
        else:
            freq_dict[i] = freq_dict[i]+1
            allowable_start_time[i-1][freq_dict[i]-1] = completion_time[i-1][(freq_dict[i]-1)-1]
            match = re.search(r'O(\d+)', process_plan_matrix[i-1][process_plan_arr[i-1]-1][freq_dict[i]-1])
            if match:
                operation_id = int(match.group(1))
            current_machine_id = operation_machine_chromosome[i-1][operation_id-1]
            process_time = operation_machine_tool_dict_list[i][operation_id][current_machine_id][0]
            if machine_engage_time_dict.get(current_machine_id) is None:
                    machine_engage_time_dict[current_machine_id] = []
            if(len(machine_engage_time_dict[current_machine_id]) == 0):
                start_time[i-1][freq_dict[i]-1] = allowable_start_time[i-1][freq_dict[i]-1]
                
            else:
                idle_s = 0
                flag = 0
                for j in range(len(machine_engage_time_dict[current_machine_id])):
                    idle_e = machine_engage_time_dict[current_machine_id][j][0]
                    if(max(allowable_start_time[i-1][freq_dict[i]-1], idle_s) + 5 <= idle_e):
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


    print(machine_engage_time_dict)
        # match = re.search(r'O(\d+)', process_plan_matrix[i-1][process_plan_arr[i-1]-1][freq_dict[i]-1])
        # if match:
        #     operation_id = int(match.group(1))
        
        # current_operation  = (i, operation_id, operation_machine_chromosome[i-1][operation_id-1], operation_machine_tool_dict_list[i-1][operation_id][operation_machine_chromosome[i-1][operation_id-1]][0])
        # print(current_operation)
    # print(chromosome)
    pass
# def operation_machine_mapping(data, num_jobs):
#     global operation_machine_dict_list

#     for i in range(num_jobs):
#         sheet_name = f'Job{i+1}'
#         sheet_data = data[sheet_name]
#         operation_machine_dict = {}

#         for index, row in sheet_data.iterrows():
#             operation_id = row['Alternative Operation']
#             operation_id = int(operation_id[1:])
            
#             alt_machines = row['Altenative Machines']

#             machine_ids = [int(machine.strip()[1:]) for machine in alt_machines.split(',')] 

#             operation_machine_dict[operation_id] = machine_ids

#         operation_machine_dict_list.append(operation_machine_dict)
        # print(operation_machine_dict)

# def machine_tool_time_mapping(data, num_jobs):
#     global operation_machine_tool_dict_list

#     for i in range(num_jobs):
#         sheet_name = f'Job{i+1}'
#         sheet_data = data[sheet_name]
#         operation_machine_tool_dict = {}

#         for index, row in sheet_data.iterrows():
#             operation_id = row['Alternative Operation']
#             operation_id = int(operation_id[1:])
            
#             alt_machines = row['Altenative Machines']

#             process_times = row["Time Taken (s)"]
#             process_times = list(map(float, process_times.split(',')))
#             machine_ids = [int(machine.strip()[1:]) for machine in alt_machines.split(',')] 

#             machine_time_dict = {}
#             i = 0
#             for mid in machine_ids:
#                 machine_time_dict[mid] = process_times[i*3:(i+1)*3]
            
#             operation_machine_tool_dict[operation_id] = machine_time_dict
        
#         operation_machine_tool_dict_list.append(operation_machine_tool_dict)
        

    # print(operation_machine_tool_dict_list)
            # print(process_times, type(process_times))
            # operation_machine_dict[operation_id] = machine_ids

        # operation_machine_dict_list.append(operation_machine_dict)
if __name__ == "__main__":
    # Call the function to load Excel data
    excel_filepath = 'D:\MTP\GA\DATASET_NEW.xlsx'  # Ensure this path is correct
    sheets_data = load_excel_data(excel_filepath)
   
    # Print or work with the separate dataframes for each sheet
    # print("DataFrame for job1:")
    # print(sheets_data['Job1'])
    
    # print("\nDataFrame for job2:")
    # print(sheets_data['Job2'])
    
    # print("\nDataFrame for job3:")
    # print(sheets_data['Job3'])

    # operation_machine_mapping(sheets_data, num_jobs)
    operation_machine_dict_list = get_alt_machines(sheets_data, num_jobs) # only required for machine chromosome
    for key in operation_machine_dict_list:
        operation_machine_dict_list

    
    # machine_tool_time_mapping(sheets_data, num_jobs)
    operation_machine_tool_dict_list = get_machine_process_time(sheets_data, num_jobs)
    # for i  in operation_machine_dict_list:
    #     print(i)
    #     print("\n")

    # print(generate_machine_chromosome(sheets=sheets_data['Job1'], job_id=1))
    # my_first_chromosome = generate_scheduling_chromosome()

    # print(my_first_chromosome["process_plan_arr"])
    # print(my_first_chromosome["scheduling_plan_arr"])
    # print(my_first_chromosome["operation_machine_chromosome"])
    # for i in range(num_jobs):
    #     print(process_plan_matrix[i][my_first_chromosome["process_plan_arr"][i]-1])
    # print("\n")

    my_demo_chromosome = {}
    my_demo_chromosome["process_plan_arr"] = np.array([6, 1, 3])
    my_demo_chromosome["scheduling_plan_arr"] = np.array([1, 3, 3, 2, 1, 3, 2, 2, 3, 0, 0, 0, 1, 1, 2, 3, 2, 0, 3, 2, 0, 1, 3, 3])
    my_demo_chromosome["operation_machine_chromosome"]= [np.array([ 2,  1,  4,  2,  7,  8,  7, 10, 11,  9,  7]), np.array([ 3,  3,  3,  2,  1,  8,  6,  7,  9,  8, 12, 12,  7,  7]), np.array([ 3,  1,  4,  2,  3,  7,  8,  7,  5,  9,  9, 13, 14,  9,  6, 11, 12])]

    for i in range(num_jobs):
        print(process_plan_matrix[i][my_demo_chromosome["process_plan_arr"][i]-1])
    print("\n")
    decode_chromosome(my_demo_chromosome)
    # Call the function to load and print the sequence dictionary
    # load_sequence()




# print(generate_initial_population())

