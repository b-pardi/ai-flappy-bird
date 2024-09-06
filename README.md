# Flappy Bird AI Visualizer with Custom FULL NEAT Implementation (No ML libraries)

This project is an implementation of the classic Flappy Bird game enhanced with an AI agent using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. The AI learns to play Flappy Bird by evolving its neural network over generations. The project includes visualization tools to monitor the evolution and structure of the neural networks over time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [CLI Arguments](#cli-arguments)
- [Project Structure](#project-structure)
- [Visualization](#visualization)
- [Pre-Trained Best Bird](#pre-trained-best-bird)
- [Logging and Debugging](#logging-and-debugging)
- [Contributing](#contributing)

## Overview

The AI in this project is built using the NEAT algorithm which I implemented myself usingly only numpy and standard Python libraries. It evolves neural networks over generations to learn how to play the game. The project visualizes the structure of these networks as they evolve, providing insights into how they change over time.

## Features

- **NeuroEvolution with NEAT:** AI learns to play Flappy Bird through evolving neural networks.
- **Dynamic Visualization:** Real-time visualization of the AI’s neural networks.
- **Logging:** Detailed logs of each generation’s best birds for debugging and insight.
- **Pre-Trained AI Bird:** A pre-trained bird model is included for quick demonstrations.

## Demo

[![Flappy Bird AI Demo](http://i3.ytimg.com/vi/GujD4FmJWK0/hqdefault.jpg)](https://youtu.be/GujD4FmJWK0)


## Installation

To run the project, follow these steps:

1. **Clone the repository:**
    `git clone https://github.com/your-username/flappy-bird-ai-neat.git`
    `cd flappy-bird-ai-neat`

2. **Set up a virtual environment (optional but recommended):**
    `python -m venv .venv`
    `source venv/bin/activate  # On Windows use `.\.venv\Scripts\activate``

3. **Install the dependencies:**
    pip install -r requirements.txt

## Usage

1. **Run the game:**
    `python main.py`
    - Optionally, set a pipe goal for the bird to reach, or run with a pretrained bird for demonstrative purposes. (See [CLI Arguments](#cli-arguments))

2. **Training the AI:**
    The AI will start training automatically. 
    *Coming Soon* To adjust training parameters or modify the AI's behavior, edit the configuration in `neat.py` or `config.json`.

3. **Visualizing the Networks:**
    The AI’s neural networks can be visualized while training using the built-in visualization tools.

## CLI Arguments

The `main.py` script accepts the following command-line arguments to customize the behavior of the game and the AI:

- **`-p` or `--pretrained`**: Runs the game using the pre-trained bird model for demonstration purposes.
Example usage: `python main.py -p`

- **`-g` or `--goal_pipes`**: Sets a target number of pipes for the leading bird to pass. Once this goal is reached, the population will die, and a new generation will be repopulated. This is useful for testing the evolution over more generations and improving the average fitness of the population, since the algorithm typically gets at least one bird to run *ad infintum* rather quickly.
Example usage: `python main.py -g x` where 'x' is an integer of pipes to reach (e.g. 20) before repopulating

## Project Structure

- **`main.py`**: The main entry point for running the game.
- **`src.neat.py`**: Contains the NEAT algorithm implementation and logic for evolving neural networks. Has the Net and Population classes.
- **`src.components.py`**: Defines the Pipe, Ground, and most importantly, Bird class that handles AI decision-making and interaction with the game. This is where inputs are fed to the network, and fitness functions are defined.
- **`src.visuals.py`**: Functions for visualizing the neural network structures and game state.
- **`logs/networks.log`**: Log file for recording the architecture of the best bird after each generation.
- **`logs/fitness_stats.log`**: Log file for recording the Fitness stats globally and across species for each generation as training progresses.
- **`logs/fitness_stats.log`**: Log file for recording the architecture of the best and worst birds after each generation.
- **`best_bird_net.pkl`**: Serialized `Net` object of the pre-trained best bird available for demonstration. This is what is loaded when running pretrained demo `python main.py -p`
- **`minimal_net.pkl`**: Serialized `Net` object of the pre-trained best bird available with **only 2 connections**. Exists to showcase how this algorithm is able to find a working solution as efficiently as possible.
- **`recent_best_bird_net.pkl`**: Serialized object of the most recently trained best bird's network recorded automatically upon closing game.

## Visualization

The project includes two primary methods for visualizing the neural network structures as they evolve:

1. Matplotlib Network Visualization: The visualization tools use networkx and matplotlib to show the structure of the neural networks in real-time. Plots are updated at the end of each generation and when the best bird has passed three pipes, allowing you to see the current working architecture of the best bird of each species each generation.
- Nodes: Represent input, hidden, and output layers of the neural network.
    - Input nodes are labeled with the type of data they receive.
- Edges: Represent the connections (weights) between nodes.
- Colors and Labels: Different colors for each type of node, with labels for node values and weights. Labels also show the input sum and activation value of each node.

2. Pygame Real-Time Visualization: In addition to the Matplotlib-based visualization, the project now includes a real-time visualization feature using Pygame. This feature displays the neural network structure of the 'best bird' (bird with the current highest fitness score) dynamically in the same window alongside the game. 
- Dynamic Updates: The network structure is updated in real-time each frame as the bird plays the game, showing the input sums (Σ) and output/activated values (σ) of each node.
- Flapping Indicator: The output node changes color when the network decides that the bird should flap, providing insight into the decision-making process.
- Connections and Weights are also updated in real-time, however because of the nature of the algorithm this occurs once every generation.
- Hidden nodes may also appear, however they algorithm pursues the simplest possible solution, so they are often unused (unconnected).

## Pre-Trained Best Bird

A pre-trained model (`best_bird_net.pkl`) is available for quick demonstrations. To load and use this bird, simply run: `python main.py -p`

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to adhere to the code style and add plenty of descriptive comments/documentation.