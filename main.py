import pygame
import sys

import src.components as comps
import src.population as popn

pygame.init()
clock = pygame.time.Clock()
population = popn.Population(500)

def check_quit_game():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

if __name__ == '__main__':
    pipe_spawn_time = 0

    while True:
        check_quit_game()

        window = comps.window
        window.fill((0,0,0))

        # draw ground
        comps.ground.draw_ground(window)

        # draw pipe
        if pipe_spawn_time <= 0:
            comps.pipes.append(comps.Pipe())
            pipe_spawn_time = comps.default_pipe_spawn_time
        pipe_spawn_time -= 1

        for pipe in comps.pipes:
            pipe.draw_pipe(window)
            pipe.update()
            if pipe.offscreen:
                comps.pipes.remove(pipe)

        if not population.is_extinct():
            population.update_live_players()
        else:
            comps.pipes.clear() # clear all existing pipes before spawning next generation
            population.natural_selection()


        clock.tick(60)
        pygame.display.flip()