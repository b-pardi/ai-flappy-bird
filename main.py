import pygame
import argparse
import sys
import matplotlib.pyplot as plt
import pickle
import src.components as components
from src.neat import Population
from src.visuals import visualize_nets, pygame_network_visualizer

pygame.init()
pygame.font.init()  # Initialize font module
font = pygame.font.SysFont('LucidaConsole', 24)
clock = pygame.time.Clock()
FPS = 120

'''WIP
- hard mode, as time goes on, shrink pipe gap
    - bird controls jump height??
- visualize live pretrained bird in pygame window (lower latency)
- more adjustable configs
- efficiency tweaks
'''

def save_best_player(population, lead_species_idx):
    best_bird = max(population.birds, key=lambda bird: bird.fitness)
    with open('logs/recent_best_bird_net.pkl', 'wb') as file:
        pickle.dump((best_bird.net, population.gen, lead_species_idx), file)

def load_best_bird_brain(path):
    with open(path, 'rb') as bird_brain:
        return pickle.load(bird_brain)


def main_game_loop(pipe_goal):
    window = pygame.display.set_mode((components.WIN_WIDTH_WITH_VISUALS, components.WIN_HEIGHT))
    pygame.display.set_caption('Training Flappy Bird...')
    population = Population(size=100, n_init_conns=1)
    pipe_spawn_time = 0
    max_generations = 100
    target_fitness = 5000
    current_plot_updated = False # will update plots each generation or if more than 3 pipes passed

    while population.gen <= max_generations:
        # check_events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                population.evaluate_fitness()
                save_best_player(population, lead_species)
                population.speciation()
                population.log_progress()
                sys.exit()
        
        window.fill((10,10,10))
        components.ground.draw_ground(window)

        # make new pipe if pipe timer hit 0
        if pipe_spawn_time <= 0:
            components.pipes.append(components.Pipe())
            pipe_spawn_time = components.default_pipe_spawn_time
        pipe_spawn_time -= 1

        # update and draw pipes moving
        for pipe in components.pipes:
            pipe.update()
            pipe.draw_pipe(window)
            if pipe.offscreen:
                components.pipes.remove(pipe)

        # update and draw birds
        population_extinct = True
        best_pipes_passed = 0
        peak_fitness = -1
        lead_species = 1
        for bird in population.birds:
            if bird.alive:
                population_extinct = False
                bird.evaluate_fitness()
                bird.think(window)
                bird.update(components.ground.ground_rect)
                bird.draw_bird(window)
                if bird.fitness > peak_fitness:
                    fittest_bird = bird
                if bird.pipes_passed > best_pipes_passed:
                    best_pipes_passed = bird.pipes_passed
                    best_bird = bird
                    for i, s in enumerate(population.species):
                        for bird in s:
                            if bird == best_bird:
                                lead_species = i + 1 # don't want to display 0 indexed

        # Display generation and best pipes passed
        gen_text = font.render(f'Generation: {population.gen}', True, (0, 150, 100))
        best_text = font.render(f'Best Pipes Passed: {best_pipes_passed}', True, (0, 150, 100))
        lead_species_text = font.render(f'Lead Species: {lead_species}', True, (0, 150, 100))
        window.blit(gen_text, (10, window.get_height() - 90))
        window.blit(best_text, (10, window.get_height() - 60))
        window.blit(lead_species_text, (10, window.get_height() - 30))

        pygame_network_visualizer(window, fittest_bird.net, components.WIN_WIDTH)

        if not current_plot_updated and best_pipes_passed > 2:
            current_plot_updated = True
            visualize_nets([best_bird.net], population.gen + 1)
        
        # stop and repopulate
        if population_extinct or (pipe_goal is not None and best_pipes_passed >= pipe_goal):
            current_plot_updated = False
            population.evaluate_fitness()
            population.speciation()
            population.log_progress()
            visualize_nets([bird.net for bird in population.species_reps], population.gen)

            population.repopulate()

            pipe_spawn_time = 0
            components.pipes.clear()
            continue

        clock.tick(FPS)
        pygame.display.flip()

def pretrained_bird_game_loop():
    window = pygame.display.set_mode((components.WIN_WIDTH_WITH_VISUALS, components.WIN_HEIGHT))
    pygame.display.set_caption('Pretrained Flappy Bird with Live Network Visualizer')
    
    # load best bird's brain
    best_bird_net, best_bird_gen, best_bird_species_idx = load_best_bird_brain("logs/best_bird_net.pkl")
    best_bird = components.Bird(best_bird_net)
    pipe_spawn_time = 0
    frame_counter = 0

    while True:
        # check_events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        window.fill((10,10,10))
        components.ground.draw_ground(window)

        # make new pipe if pipe timer hit 0
        if pipe_spawn_time <= 0:
            components.pipes.append(components.Pipe())
            pipe_spawn_time = components.default_pipe_spawn_time
        pipe_spawn_time -= 1

        # update and draw pipes moving
        for pipe in components.pipes:
            pipe.update()
            pipe.draw_pipe(window)
            if pipe.offscreen:
                components.pipes.remove(pipe)

        # update and draw birds
        best_pipes_passed = 0
        if best_bird.alive:
            best_bird.think(window)
            best_bird.update(components.ground.ground_rect)
            best_bird.draw_bird(window)
            if best_bird.pipes_passed > best_pipes_passed:
                best_pipes_passed = best_bird.pipes_passed
                best_bird = best_bird
                    
        # Display generation and best pipes passed
        gen_text = font.render(f'Generation: {best_bird_gen}', True, (0, 150, 100))
        best_text = font.render(f'Best Pipes Passed: {best_pipes_passed}', True, (0, 150, 100))
        lead_species_text = font.render(f'Lead Species: {best_bird_species_idx}', True, (0, 150, 100))
        window.blit(gen_text, (10, window.get_height() - 90))
        window.blit(best_text, (10, window.get_height() - 60))
        window.blit(lead_species_text, (10, window.get_height() - 30))

        pygame_network_visualizer(window, best_bird.net, components.WIN_WIDTH)
        
        if not best_bird.alive:
            pygame.quit()
            sys.exit()

        frame_counter += 1
        clock.tick(FPS)
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="AI Flappy bird")
    parser.add_argument('-p', '--pretrained', action='store_true', help="Use pretrained bird for demonstration")
    parser.add_argument('-g', '--goal_pipes', type=int, default=None, help="Set a goal for pipes passed for the lead bird, which will trigger repopulation when reached")
    args = parser.parse_args()

    if args.pretrained:
        pretrained_bird_game_loop()
    else:
        main_game_loop(args.goal_pipes)

if __name__ == '__main__':
    main()