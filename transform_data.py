def get_alt_machines(data, num_jobs):
    all_alt_machines_dict  = {}
    for i in range(num_jobs):
        sheet_name = f'Job{i+1}'
        sheet_data = data[sheet_name]

        alt_machines_dict = {}

        for index, row in sheet_data.iterrows():
            operation_id = row['Alternative Operation']
            operation_id = int(operation_id[1:])

            alt_machines = row['Altenative Machines']

            machine_ids = [int(machine.strip()[1:]) for machine in alt_machines.split(',')]

            alt_machines_dict[operation_id] = machine_ids

        all_alt_machines_dict[i+1] = alt_machines_dict 
    
    return all_alt_machines_dict

def get_machine_process_time(data, num_jobs):
    process_time_matrix = {}

    for i in range(num_jobs):
        sheet_name = f'Job{i+1}'
        sheet_data = data[sheet_name]

        operation_time_matrix = {}

        for index, row in sheet_data.iterrows():
            operation_id = row['Alternative Operation']
            operation_id = int(operation_id[1:])

            alt_machines = row['Altenative Machines']
            machine_ids = [int(machine.strip()[1:]) for machine in alt_machines.split(',')] 

            process_times = row["Time Taken (s)"]
            process_times = list(map(float, process_times.split(',')))

            machine_tool_time_matrix = {}

            j = 0
            for m_id in machine_ids:
                machine_tool_time_matrix[m_id] = process_times[j*3: (j+1)*3]

            operation_time_matrix[operation_id] = machine_tool_time_matrix
        
        process_time_matrix[i+1] = operation_time_matrix

    return process_time_matrix


def get_machine_energy(data, num_jobs):
    energy_matrix = {}

    for i in range(num_jobs):
        sheet_name = f'Job{i+1}'
        sheet_data = data[sheet_name]

        operation_energy_matrix = {}

        for index, row in sheet_data.iterrows():
            operation_id = row['Alternative Operation']
            operation_id = int(operation_id[1:])

            alt_machines = row['Altenative Machines']
            machine_ids = [int(machine.strip()[1:]) for machine in alt_machines.split(',')] 

            energy = row["Energy Consumed (KJ)"]
            energy = list(map(float, energy.split(',')))

            machine_tool_energy_matrix = {}

            j = 0
            for m_id in machine_ids:
                machine_tool_energy_matrix[m_id] = energy[j*3: (j+1)*3]

            operation_energy_matrix[operation_id] = machine_tool_energy_matrix
        
        energy_matrix[i+1] = operation_energy_matrix

    return energy_matrix