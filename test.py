def decode_chromosome(chromosome, num_jobs, len_plan_max, process_plan_matrix, process_time_matrix):
    
    schedule_plan_arr = chromosome.schedule_plan_gene
    process_plan_arr = chromosome.process_plan_gene
    operation_machine_chromosome = chromosome.operation_machine_gene

    print("operationMC", operation_machine_chromosome)
    freq_dict = {}
    machine_engage_time_dict = {}
    machine_last_completion_time_arr = np.zeros(16)
    allowable_start_time = np.zeros((num_jobs, len_plan_max))
    start_time = np.zeros((num_jobs, len_plan_max))
    completion_time = np.zeros((num_jobs, len_plan_max))
    # print(schedule_plan_arr)
    for i in schedule_plan_arr:
        if i == 0:
            continue
        if freq_dict.get(i) is None:
            freq_dict[i] = 1
            # print(i-1, process_plan_arr[i-1]-1, freq_dict[i]-1)
            match = re.search(r'O(\d+)', process_plan_matrix[i-1][process_plan_arr[i-1]-1][freq_dict[i]-1])
            if match:
                operation_id = int(match.group(1))
            allowable_start_time[i-1][freq_dict[i]-1] = 0
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

    machine_engage_time_dict, np.max(machine_last_completion_time_arr)
    print(machine_last_completion_time_arr)
    print(machine_engage_time_dict)
