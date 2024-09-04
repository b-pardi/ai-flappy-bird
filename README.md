# Flappy Bird AI with Custom FULL NEAT Implementation (No ML libraries)

This project is an implementation of the classic Flappy Bird game enhanced with an AI agent using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. The AI learns to play Flappy Bird by evolving its neural network over generations. The project includes visualization tools to monitor the evolution and structure of the neural networks over time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
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
    - Optionally, set a pipe goal for the bird to reach, upon which the population will die and repopulate, in effort to see morer generations and not run infinitely with just 1 bird. `python main.py -g x` where x is an integer of pipes to reach (e.g. 20)

2. **Training the AI:**
    The AI will start training automatically. 
    *Coming Soon* To adjust training parameters or modify the AI's behavior, edit the configuration in `neat.py` or `config.json`.

3. **Visualizing the Networks:**
    The AI’s neural networks can be visualized while training using the built-in visualization tools.

## Project Structure

- **`main.py`**: The main entry point for running the game.
- **`src.neat.py`**: Contains the NEAT algorithm implementation and logic for evolving neural networks. Has the Net and Population classes.
- **`src.components.py`**: Defines the Pipe, Ground, and most importantly, Bird class that handles AI decision-making and interaction with the game. This is where inputs are fed to the network, and fitness functions are defined.
- **`src.visuals.py`**: Functions for visualizing the neural network structures and game state.
- **`pipes.py`**: Handles the game logic for pipes.
- **`components/`**: Contains additional game components and assets.
- **`logs/networks.log`**: Log file for recording the architecture of the best bird after each generation.
- **`logs/fitness_stats.log`**: Log file for recording the Fitness stats globally and across species for each generation as training progresses.
- **`logs/fitness_stats.log`**: Log file for recording the architecture of the best and worst birds after each generation.
- **`best_bird_net.pkl`**: Serialized `Net` object of the pre-trained best bird available for demonstration. This is what is loaded when running pretrained demo `python main.py -p`
- **`recent_best_bird_net.pkl`**: Serialized object of the most recently trained best bird's network recorded automatically upon closing game.

## Visualization

The visualization tools use `networkx` and `matplotlib` to show the structure of the neural networks in real-time. Plots are updated at the end of each generation, and once the best bird has passed 3 pipes to see the current working architecture.

If using the pretrained, plot updates every 60 frames to see real-time decision making

- **Nodes:** Represent input, hidden, and output layers of the neural network.
    - Input nodes are labelled with the type of data coming into it.
- **Edges:** Represent the connections (weights) between nodes.
- **Colors and Labels:** Different colors for each type of node, with labels for node values and weights. Labels also show the input sum and activation value of each node.

## Pre-Trained Best Bird

A pre-trained model (`best_bird_net.pkl`) is available for quick demonstrations. To load and use this bird, simply run: `python main.py -p`

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to adhere to the code style and include tests for any new features.