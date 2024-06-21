import pygame
import random

import src.net as net

win_height = 720
win_width = 550
default_pipe_spawn_time = 200
bird_x = 50

class Ground:
    ground_level = win_height-100
    line_width = 5

    def __init__(self):
        self.x, self.y = 0, Ground.ground_level
        self.rect = pygame.Rect(self.x, self.y, win_width, Ground.line_width)

    def draw_ground(self, draw_window):
        pygame.draw.rect(draw_window, (255,255,255), self.rect)

class Pipe:
    width = 15
    opening_size = 100

    def __init__(self):
        # spawn new pipes on the far right of window
        self.x = win_width
        self.top_height = random.randint(50, Ground.ground_level - self.opening_size - 50)
        self.bottom_height = Ground.ground_level - self.top_height - self.opening_size

        self.bottom_rect, self.top_rect = pygame.Rect(0,0,0,0), pygame.Rect(0,0,0,0)
        self.passed = False
        self.offscreen = False

    def draw_pipe(self, draw_window):
        self.bottom_rect = pygame.Rect(self.x, Ground.ground_level - self.bottom_height, self.width, self.bottom_height)
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        
        pygame.draw.rect(draw_window, (200,200,200), self.bottom_rect)
        pygame.draw.rect(draw_window, (200,200,200), self.top_rect)

    def update(self):
        self.x -= 1
        if self.x + Pipe.width < Bird.size + Bird.init_x_loc:
            self.passed = True

        if self.x + Pipe.width < - self.width:
            self.offscreen = True

class Bird:
    size = 20
    init_x_loc = 50
    init_y_loc = 200
    vel_incr = 0.25
    vel_cap = 5
    flap_thresh = 0.73 # threshold value that if guessed btwn 0 and 1, will flap if above

    def __init__(self):
        self.x, self.y = self.init_x_loc, self.init_y_loc
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        # random colors for multiple generations
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        # physics
        self.vel = 0
        self.flap = False
        self.alive = True

        # AI vars
        self.decision = None
        self.vision = [0.5, 1, 0.5]
        self.n_inputs = 3
        self.brain = net.Net(self.n_inputs)
        self.brain.generate_net()

    def draw_bird(self, draw_window):
        pygame.draw.rect(draw_window, self.color, self.rect)

    def ground_collision(self, ground_rect):
        return pygame.Rect.colliderect(self.rect, ground_rect)
    
    def sky_collision(self):
        return self.rect.y < self.size
    
    def pipe_collision(self):
        for pipe in pipes:
            return pygame.Rect.colliderect(self.rect, pipe.top_rect) or pygame.Rect.colliderect(self.rect, pipe.bottom_rect)

    def update(self, ground_rect):
        if self.ground_collision(ground_rect) or self.pipe_collision():
            self.alive = False
            self.flap = False
            self.vel = 0
        else:
            self.vel += self.vel_incr
            self.rect.y += self.vel
            if self.vel > 5:
                self.vel = 5

    def bird_flap(self):
        if not self.flap and not self.sky_collision():
            self.flap = True
            self.vel = -5
        if self.vel >= 3:
            self.flap = False

    @staticmethod
    def closest_pipe():
        for pipe in pipes:
            if not pipe.passed:
                return pipe
            
    def look(self):
        if pipes:
            # line to top pipe
            player_y = self.rect.center[1]
            player_x = self.rect.center[0]
            bottom_of_top_pipe = self.closest_pipe().top_rect.bottom
            left_of_closest_pipe = self.closest_pipe().x
            top_of_bottom_pipe = self.closest_pipe().bottom_rect.top
            self.vision[0] = max(0, player_y - bottom_of_top_pipe) / Ground.ground_level
            self.vision[1] = max(0, left_of_closest_pipe - player_x) / Ground.ground_level
            self.vision[2] = max(0, top_of_bottom_pipe - player_y) / Ground.ground_level


            pygame.draw.line(window, self.color, self.rect.center, (player_x, bottom_of_top_pipe))
            pygame.draw.line(window, self.color, self.rect.center, (left_of_closest_pipe, player_y))
            pygame.draw.line(window, self.color, self.rect.center, (player_x, top_of_bottom_pipe))

    # AI functions
    def think(self):
        # to flap or not to flap, that is the question
        self.decision = self.brain.ff(self.vision)
        if self.decision > self.flap_thresh:
            self.bird_flap()

window = pygame.display.set_mode((win_width, win_height))

ground = Ground()
pipes = []
