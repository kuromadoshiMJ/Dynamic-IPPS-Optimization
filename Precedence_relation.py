import pandas as pd
import networkx as nx
from itertools import permutations

# Load the precedence table from the Excel file
def load_precedence_table(file_path, sheet_name):
    print("Loading data from Excel file...")
    df = pd.read_excel(file_path, sheet_name= sheet_name)
    print(f"Data loaded:\n{df.head()}")
    return df

# Create the graph based on the precedence table
def create_graph(df):
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        print(f"Adding edge from {row['From']} to {row['To']}")
        G.add_edge(row['From'], row['To'])
    
    return G

# Check if a permutation is a valid path according to the precedence graph
def is_valid_path(graph, path):
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True

# Generate all valid sequences based on the graph
def find_all_paths(graph):
    nodes = list(graph.nodes)
    all_paths = []
    print(f"Nodes: {nodes}")

    # Generate all permutations of nodes
    for perm in permutations(nodes):
        # Check the sequence start condition
        if perm[0] not in {'F1', 'F2'}:
            continue
        
        # Check the no consecutive F1-F2 or F2-F1 condition
        if 'F1' in perm and 'F2' in perm:
            f1_index = perm.index('F1')
            f2_index = perm.index('F2')
            if abs(f1_index - f2_index) == 1:
                continue
        
        # Check for precedence constraints
        if is_valid_path(graph, perm):
            all_paths.append(perm)
    
    return all_paths

# Print all possible valid sequences
def print_all_paths(paths):
    if not paths:
        print("No valid paths found.")
    else:
        for path in paths:
            print(" → ".join(path))

def Precedence(file_path, sheet_name):
    # Load precedence table
    df = load_precedence_table(file_path, sheet_name)
    
    # Create the precedence graph
    graph = create_graph(df)
    
    # Find all valid paths
    paths = find_all_paths(graph)
    
    # Print all possible valid sequences
    print_all_paths(paths)
    return paths

if __name__ == "__main__":
    # Provide the path to your Excel file
    file_path = 'precedence_table.xlsx'
    (file_path)
