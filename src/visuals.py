import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.neat import NodeType

def visualize_nets(nets, gen):
    """
    Visualizes multiple neural network structures side by side with different colors for each layer.
    
    Args:
    - nets (list of Net): list of network objects containing nodes and connections.
    - fig_name (str): Title for the figure window.
    """
    fig = plt.gcf()
    fig.clf()
    
    n_birds = len(nets)
    fig, axes = plt.subplots(1, n_birds, figsize=(6 * n_birds, 6), num=fig.number)
    fig.suptitle(f"Generation: {gen} Best Networks")
    if n_birds == 1:
        axes = [axes]

    # Iterate over each network and plot it on a separate subplot
    for idx, (net, ax) in enumerate(zip(nets, axes)):
        G = nx.DiGraph()

        # Extract node types
        input_nodes = net.nodes[net.nodes['type'] == NodeType.INPUT.value]
        hidden_nodes = net.nodes[net.nodes['type'] == NodeType.HIDDEN.value]
        output_nodes = net.nodes[net.nodes['type'] == NodeType.OUTPUT.value]

        # Determine the x-coordinates for each layer
        input_x = 0
        hidden_x = 1
        output_x = 2

        # Calculate y-coordinates for vertical centering
        input_y = np.linspace(0, 1, len(input_nodes))[::-1]  # Vertically center input nodes
        hidden_y = np.linspace(0.7, 1.2 + len(hidden_nodes) // 5, len(hidden_nodes)) if len(hidden_nodes) > 0 else [0]  # Center hidden nodes
        output_y = np.linspace(0, 1, len(output_nodes))  # Vertically center output nodes

        # Initialize positions dictionary
        pos = {}

        # Add nodes to the graph with their respective positions
        for i, node in enumerate(input_nodes):
            G.add_node(node['id'], pos=(input_x, input_y[i]))
            pos[node['id']] = (input_x, input_y[i])  # Store positions in the dictionary

        for i, node in enumerate(hidden_nodes):
            G.add_node(node['id'], pos=(hidden_x, hidden_y[i]))
            pos[node['id']] = (hidden_x, hidden_y[i])  # Store positions in the dictionary

        for i, node in enumerate(output_nodes):
            G.add_node(node['id'], pos=(output_x, output_y[i]))
            pos[node['id']] = (output_x, output_y[i])  # Store positions in the dictionary

        # Add edges to the graph based on connections
        for conn in net.connections:
            if conn['enabled']:
                in_node_id = conn['in_node_id']
                out_node_id = conn['out_node_id']
                if in_node_id in G.nodes and out_node_id in G.nodes:  # Check if nodes exist in the graph
                    G.add_edge(in_node_id, out_node_id, weight=conn['weight'])

        # Draw the graph with positions
        edge_labels = nx.get_edge_attributes(G, 'weight')

        # Draw nodes with different colors
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes['id'], node_color='lightblue', node_size=500, label='Input Nodes', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes['id'], node_color='lightgreen', node_size=500, label='Hidden Nodes', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes['id'], node_color='lightcoral', node_size=500, label='Output Nodes', ax=ax)

        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=15, arrowstyle='-|>', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight')}, font_color='black', font_size=10, ax=ax)

        # labels for nodes
        node_labels_dict = {}
        node_info_dict = {}
        input_labels_dict = {0: "Top Pipe Y Dist", 1: "Closest Pipe X Dist", 2: "Bottom Pipe Y Dist", 4: "Bias"}
        for node in net.nodes:
            node_labels_dict[node['id']] = f"{node['id']}"
            node_info_dict[node['id']] = f"Σ: {node['input_sum']:.2f}\nO: {node['output_value']:.2f}"
            
        offset_lower_y_pos = {node_id: (x, y - 0.1) for node_id, (x, y) in pos.items()}  # Apply slight y-offset
        offset_lower_x_pos = {node_id: (x- 0.1, y) for node_id, (x, y) in pos.items()}  # Apply slight x-offset
        
        nx.draw_networkx_labels(G, pos, labels=node_labels_dict, font_size=10, font_color='black', ax=ax) # draw node labels
        nx.draw_networkx_labels(G, offset_lower_y_pos, labels=node_info_dict, font_size=8, font_color='black', ax=ax) # draw node information

        # label input nodes
        for node_id, label in input_labels_dict.items():
            x, y = offset_lower_x_pos[node_id]
            ax.text(x, y, label, fontsize=8, color='black', ha='right', va='center')

        ax.set_title(f'Species {idx + 1}')
        ax.set_xlim(input_x - 0.5, output_x + 0.4)  # Adds padding around the plot to avoid cropping labels
        ax.set_ylim(min(input_y) - 0.25, max(max(input_y), max(hidden_y)) + 0.1)  # Adjust limits to center everything nicely
        ax.axis('off')  # Turn off the axis to remove the plot border
    
    plt.tight_layout()
    plt.draw()  # Redraw the current figure
    plt.pause(0.001)  # Non-blocking show

