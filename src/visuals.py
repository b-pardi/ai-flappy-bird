import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pygame
from enum import Enum

from src.neat import NodeType
import src.components as components

class VisualizerType(Enum):
    MATPLOTLIB = 'matplotlib'
    PYGAME = 'pygame'
    DISABLED = 'disabled'

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
        output_y = np.linspace(0, 1, len(output_nodes))

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

def pygame_network_visualizer(screen, net, x_offset):
    """
    Draws the neural network visualization directly on the main Pygame screen.
    
    Args:
    - screen: The main Pygame screen.
    - net (Net): The network object containing nodes and connections.
    - x_offset (int): Horizontal offset for the drawing area.
    - y_offset (int): Vertical offset for the drawing area.
    """

    font = pygame.font.SysFont('LucidaConsole', 18)
    bigger_font = pygame.font.SysFont('LucidaConsole', 24)

    colors = {
        'input': (173, 216, 230),   # Light blue
        'hidden': (144, 238, 144),  # Light green
        'output': (240, 128, 128),  # Light coral
        'text': (200, 200, 200),    # Light gray
        'id_text': (0, 0, 0),       # Black
        'edge': (100, 100, 100),    # Gray
        'highlight': (0, 250, 0),   # Green for flapping state
        'separator': (100,100,100), # bar to separate plot and game
        'title': (200, 0, 220)      # title color
    }

    # draw rectangle to separate plot from game
    separator_rect = pygame.Rect(x_offset, 0, components.Pipe.width, components.WIN_HEIGHT)
    pygame.draw.rect(screen, colors['separator'], separator_rect)

    # Title text 875
    title_text_surface = bigger_font.render("Fittest Bird's Network", True, colors['title'])
    title_loc = (components.WIN_WIDTH_WITH_VISUALS - ((components.WIN_WIDTH_WITH_VISUALS - components.WIN_WIDTH) // 2) - (title_text_surface.get_width() // 2) + components.Pipe.width, 10)
    screen.blit(title_text_surface, title_loc)

    # Determine positions for the nodes
    input_nodes = net.nodes[net.nodes['type'] == NodeType.INPUT.value]
    hidden_nodes = net.nodes[net.nodes['type'] == NodeType.HIDDEN.value]
    output_nodes = net.nodes[net.nodes['type'] == NodeType.OUTPUT.value]

    input_x, hidden_x, output_x = x_offset + 100, x_offset + 325, x_offset + 550
    input_y = np.linspace(100, components.WIN_HEIGHT-160, len(input_nodes))
    hidden_y = np.linspace(components.WIN_HEIGHT - 80, components.WIN_HEIGHT // 2, len(hidden_nodes))[::-1]
    output_y = np.linspace(70, components.WIN_HEIGHT-400, len(output_nodes))

    node_positions = {}
    for i, node in enumerate(input_nodes):
        node_positions[node['id']] = (input_x, input_y[i])

    for i, node in enumerate(hidden_nodes):
        node_positions[node['id']] = (hidden_x, hidden_y[i])

    for i, node in enumerate(output_nodes):
        node_positions[node['id']] = (output_x, output_y[i])

    # Draw connections (edges)
    for conn in net.connections:
        if conn['enabled']:
            try: # ensure valid connection is displayed
                in_pos = node_positions[conn['in_node_id']]
                out_pos = node_positions[conn['out_node_id']]
            except KeyError as ke:
                continue
            pygame.draw.line(screen, colors['edge'], in_pos, out_pos, 2)

            # Draw weight value
            mid_pos = ((in_pos[0] + out_pos[0]) // 2, (in_pos[1] + out_pos[1]) // 2)
            weight_text = f"{conn['weight']:.2f}"
            text_surface = font.render(weight_text, True, colors['text'])
            screen.blit(text_surface, mid_pos)

    # Draw nodes
    input_labels_dict = {0: "Top Pipe\nY Dist", 1: "Closest Pipe\nX Dist", 2: "Bottom Pipe\nY Dist", 4: "Bias"}
    for node in net.nodes:
        pos = node_positions[node['id']]
        node_color = colors['input'] if node['type'] == NodeType.INPUT.value else colors['hidden'] if node['type'] == NodeType.HIDDEN.value else colors['output']

        # Change output node color based on output value
        if node['type'] == NodeType.OUTPUT.value and node['output_value'] > 0.5:  # Assume output > 0.5 triggers flap
            node_color = colors['highlight']  # Red color for flap

        pygame.draw.circle(screen, node_color, pos, 20)

        # Draw node ID inside node
        node_id_text = bigger_font.render(str(node['id']), True, colors['id_text'])
        text_rect = node_id_text.get_rect(center=pos)  # Center text inside the node
        screen.blit(node_id_text, text_rect)

        # Draw labels, input sum, and output value centered below each node
        input_sum = f"Σ: {node['input_sum']:.2f}"
        output_value = f"σ: {node['output_value']:.2f}"
        input_text_surface = font.render(input_sum, True, colors['text'])
        output_text_surface = font.render(output_value, True, colors['text'])

        # Calculate position for centered text below node
        input_text_pos = (pos[0] - input_text_surface.get_width() // 2, pos[1] + 25)
        output_text_pos = (pos[0] - output_text_surface.get_width() // 2, pos[1] + 46) 
        screen.blit(input_text_surface, input_text_pos)
        screen.blit(output_text_surface, output_text_pos)

        if node['id'] in input_labels_dict.keys(): # draw input node labels
            # pygame is annoying and doesn't render new lines, so they must be drawn seperately
            text = input_labels_dict[node['id']]
            lines = text.split('\n')
            y_offset = 40
            for i, line in enumerate(lines):
                label_text_surface = font.render(line, True, colors['text'])
                label_text_pos = (pos[0] - label_text_surface.get_width() // 2, pos[1] - y_offset)
                screen.blit(label_text_surface, label_text_pos)
                y_offset += 20
           