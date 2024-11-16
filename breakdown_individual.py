import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
class Chromosome:
    def __init__(self, num_jobs, len_plan_max, num_operations_arr, bd_start = None, bd_end = None, bd_machine = None) -> None:
        self. process_plan_gene = np.zeros(num_jobs, dtype=int)
        self. schedule_plan_gene = np.zeros(num_jobs * len_plan_max, dtype=int)
        self. operation_machine_gene = [] 
        self. operation_tool_gene = []
        self. total_energy = 0
        self. make_span = 0
        self. objectives = None
        
        self.rank = None
        self.crowding_distance = 0
        self.domination_count = 0
        self.dominated_solutions = []

        self.bd_start = bd_start
        self.bd_end = bd_end
        self.bd_machine = bd_machine
        if bd_machine is not None: 
            self.machine_engage_time_dict = {
                                                int(self.bd_machine):[[self.bd_start, self.bd_end, -1, -1]]
                                            }
        else:
            self.machine_engage_time_dict = {}
        for i in range(num_jobs):
            self.operation_machine_gene.append(np.zeros(num_operations_arr[i], dtype=int))
        for i in range(num_jobs):
            self.operation_tool_gene.append(np.zeros(num_operations_arr[i], dtype=int))

    def encode_chromosome(self, num_jobs, num_process_plan_arr, len_process_plan_arr, num_operations_arr, len_plan_max, alt_machines_dict) -> None:
        self.process_plan_gene = (np.random.randint(0, num_process_plan_arr)) + 1

        schedule_plan_list = []
        for i in range(1, num_jobs+1):
            schedule_plan_list.extend([i] * len_process_plan_arr[i-1])

        num_zeros = (num_jobs * len_plan_max) - len(schedule_plan_list)
        schedule_plan_list.extend([0] * num_zeros)
        schedule_plan_arr = np.array(schedule_plan_list) 

        self.schedule_plan_gene = np.random.permutation(schedule_plan_arr)

        self.operation_machine_gene = []
        self.operation_tool_gene = []
        for i in range(num_jobs):
            job_id = i+1
            machine_plan = self.generate_machine_gene(num_operations_arr, job_id, alt_machines_dict)
            tool_plan = self.generate_tool_gene(num_operations_arr, job_id)
            self.operation_machine_gene.append(machine_plan)
            self.operation_tool_gene.append(tool_plan)

    def generate_machine_gene(self, num_operations_arr, job_id, alt_machines_dict):
        machine_plan_arr = np.zeros(num_operations_arr[job_id - 1], dtype=int)
        
        operation_machine_dict = alt_machines_dict[job_id]
        
        for i in range(num_operations_arr[job_id -1]):
            machine_plan_arr[i] = np.random.choice(operation_machine_dict[i+1])

        return machine_plan_arr
    
    def generate_tool_gene(self, num_operations_arr, job_id):
        tool_plan_arr = np.zeros(num_operations_arr[job_id-1], dtype = int)

        for i in range(num_operations_arr[job_id-1]):
            tool_plan_arr[i] = np.random.choice([0, 1, 2])
        
        return tool_plan_arr
    
    def decode_chromosome(self, num_jobs, num_machines, len_plan_max, process_plan_matrix, process_time_matrix, energy_matrix):
        # print("operationMC", self.operation_machine_gene)
        freq_dict = {}
        # self.machine_engage_time_dict = {}
        machine_last_completion_time_arr = np.zeros(num_machines)
        allowable_start_time = np.zeros((num_jobs, len_plan_max))
        start_time = np.zeros((num_jobs, len_plan_max))
        completion_time = np.zeros((num_jobs, len_plan_max))

        if(self.bd_end is not None):
            machine_last_completion_time_arr[self.bd_machine-1] = self.bd_end
        # self.machine_engage_time_dict[self.bd_machine] = [[self.bd_start, self.bd_end, -1, -1]]
        for i in self.schedule_plan_gene:
            if i == 0:
                continue
            if freq_dict.get(i) is None:
                freq_dict[i] = 1
                allowable_start_time[i-1][freq_dict[i]-1] = 0
                    
            else:
                freq_dict[i] = freq_dict[i]+1
                allowable_start_time[i-1][freq_dict[i]-1] = completion_time[i-1][(freq_dict[i]-1)-1]
            
            match = re.search(r'O(\d+)', process_plan_matrix[i-1][self.process_plan_gene[i-1]-1][freq_dict[i]-1])
            if match:
                operation_id = int(match.group(1))
            current_machine_id = self.operation_machine_gene[i-1][operation_id-1]

            # print("len", len(self.operation_tool_gene))
            # for j in self.operation_tool_gene:
            #     print(j)

            # print(i-1, operation_id-1)
            # print(self.operation_tool_gene[i-1])
            # print(self.oper)
            
            current_tool_id = self.operation_tool_gene[i-1][operation_id-1]

            # print(i, operation_id, current_machine_id, current_tool_id)
            # print(i, operation_id, current_machine_id, 0)
            process_time = process_time_matrix[i][operation_id][current_machine_id][current_tool_id]
            energy = energy_matrix[i][operation_id][current_machine_id][current_tool_id]
            self.total_energy += energy
            if self.machine_engage_time_dict.get(current_machine_id) is None:
                    self.machine_engage_time_dict[current_machine_id] = []
            if(len(self.machine_engage_time_dict[current_machine_id]) == 0):
                start_time[i-1][freq_dict[i]-1] = allowable_start_time[i-1][freq_dict[i]-1]
                
            else:
                idle_s = 0
                flag = 0
                for j in range(len(self.machine_engage_time_dict[current_machine_id])):
                    idle_e = self.machine_engage_time_dict[current_machine_id][j][0]
                    if(max(allowable_start_time[i-1][freq_dict[i]-1], idle_s) + process_time <= idle_e):
                        start_time[i-1][freq_dict[i]-1] = max(allowable_start_time[i-1][freq_dict[i]-1], idle_s)
                        flag = 1
                        break
                    idle_s = self.machine_engage_time_dict[current_machine_id][j][1]
                    
                if(flag == 0):
                    start_time[i-1][freq_dict[i]-1] = max(allowable_start_time[i-1][freq_dict[i]-1], machine_last_completion_time_arr[current_machine_id-1])

            completion_time[i-1][freq_dict[i]-1] = start_time[i-1][freq_dict[i] - 1] + process_time
            self.machine_engage_time_dict[current_machine_id].append([start_time[i-1][freq_dict[i]-1], completion_time[i-1][freq_dict[i]-1], i, operation_id]) 
            self.machine_engage_time_dict[current_machine_id].sort()
            machine_last_completion_time_arr[current_machine_id-1] = max(machine_last_completion_time_arr[current_machine_id-1],  completion_time[i-1][freq_dict[i]-1])

        self.make_span = machine_last_completion_time_arr.max()
        
        self.objectives = np.array([self.make_span, self.total_energy])
        
    def plot_gantt_chart(self):
        data =[]

        colo_mapping = {
            1 : 'blue',
            2 : 'green',
            3 : 'orange'
        }

        for machine_id, operations in self.machine_engage_time_dict.items():
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