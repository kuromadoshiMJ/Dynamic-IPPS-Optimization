import networkx as nx

# Create the graph 

job_graph = [nx.DiGraph(), nx.DiGraph(), nx.DiGraph()]
G1 = nx.DiGraph()
G2 = nx.DiGraph()
G3 = nx.DiGraph()

# Add edges for OR node 
job_graph[0].add_edges_from([("S", "O1"), ("O1", "O3"), ("O3", "OR2"), ("OR2", "O2"), ("O2", "OJ2"),
                                                            ("OR2", "O7"), ("O7", "OJ2"),("OJ2", "OR3"), ("OR3", "O4"), ("O4", "OJ3"),
                                                                                                        ("OR3", "O8"),  ("O8", "OJ3"), ("OJ3", "O5"), ("O5", "E"),  
                    ("S", "O11"), ("O11", "O3_2"), ("O3_2", "OR4"), ("OR4", "O6"), ("O6", "OJ4"),
                                                                    ("OR4", "O10"), ("O10", "OJ4"), ("OJ4", "O9"), ("O9", "O5_2"), ("O5_2", "E")])

job_graph[1].add_edges_from([("S", "O1"), ("O1", "O3"), ("O3", "OR2"), ("OR2", "O4"), ("O4", "OJ2"),
                                                            ("OR2", "O9"), ("O9", "OJ2"), ("OJ2", "O5"), ("O5", "OR3"), ("OR3", "O2"), ("O2", "OJ3"),
                                                                                                                        ("OR3", "O8"), ("O8", "OJ3"), ("OJ3", "O6"), ("O6", "E"),  
                    ("S", "O14"), ("O14", "O3_2"), ("O3_2", "O6_2"), ("O6_2", "OR4"), ("OR4", "O7"), ("O7", "OJ4"),
                                                                                ("OR4", "O13"), ("O13", "OJ4"), ("OJ4", "O11"), ("O11", "OR5"), ("OR5", "O10"), ("O10", "OJ5"),
                                                                                                                                                ("OR5", "O12"), ("O12", "OJ5"), ("OJ5", "E")])

job_graph[2].add_edges_from([("S", "O1"), ("O1", "O3"), ("O3", "O4"), ("O4", "O10"), ("O10", "O5"), ("O5", "OR2"), ("OR2", "O2"), ("O2", "OJ2"),
                                                                                                        ("OR2", "O9"), ("O9", "OJ2"), ("OJ2", "O13"), ("O13", "O6"), ("O6", "E"),  
                    ("S", "O15"), ("O15", "O3_2"), ("O3_2", "O12"), ("O12", "O7"), ("O7", "OR3"), ("OR3", "O8"), ("O8", "OJ3"),
                                                                                                ("OR3", "O14"), ("O14", "OJ3"), ("OJ3", "O4_2"), ("O4_2", "O17"), ("O17", "O16"), ("O16", "E")])

# Helper function to traverse the graph and get all process plans

def dfs_traverse(G, node, path, all_paths):
    # If we reach the 'End' node, record the current path as a valid process plan
    if node == "E":
        path.pop()
        all_paths.append(path.copy())
        path.append("E")
        return
    
    # Traverse through all neighbors of the current node
    for neighbor in G.neighbors(node):
        if neighbor.startswith("OR") or neighbor.startswith("OJ"):
            dfs_traverse(G, neighbor, path, all_paths)
        else:
            path.append(neighbor)
            dfs_traverse(G, neighbor,path, all_paths)
            path.pop()


def get_process_plan_matrix():
    process_plan_matrix = []


    for i in range(3):
        all_process_plans = []
        path = []
        dfs_traverse(job_graph[i], "S", path, all_process_plans)
        process_plan_matrix.append(all_process_plans)

    return process_plan_matrix

# dfs_traverse(G3, "S", path, all_process_plans)

# Print all process plans

# for p in process_plan_matrix:
#     for i in p:
#         print(i)
#     print("\n\n--")

# print(process_plan_matrix[2][3])
# print(process_plan_matrix)
# print("All possible process plans:")
# for idx, plan in enumerate(all_process_plans, 1):
#     print(f"Plan {idx}: {' -> '.join(plan)}")
