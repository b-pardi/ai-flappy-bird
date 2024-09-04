import pygame
import random

WIN_WIDTH = 550
WIN_HEIGHT = 720

clock = pygame.time.Clock()
window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

default_pipe_spawn_time = 240
pipes = []

class Ground:
    ground_level = WIN_HEIGHT - 100 # ground is 100 pixels above window bottom
    line_width = 5 # width of drawn line for ground

    def __init__(self):
        self.x = 0
        self.y = Ground.ground_level
        self.ground_rect = pygame.Rect(self.x, self.y, WIN_WIDTH, Ground.line_width)

    def draw_ground(self, window):
        pygame.draw.rect(window, (220,220,220), self.ground_rect)

class Pipe:
    width = 15 # width of pipe to draw
    pipe_gap = 100 # distance between pipe for bird to flap through
    min_pipe_part_length = 100 # minimum length of either pipe part

    def __init__(self):
        self.x = WIN_WIDTH
        self.top_part_length = random.randint(Pipe.min_pipe_part_length, Ground.ground_level - Pipe.min_pipe_part_length - Pipe.pipe_gap)
        self.bottom_part_length = Ground.ground_level - Pipe.pipe_gap - self.top_part_length
        self.passed = False
        self.offscreen = False
        self.color = (220,220,220)

    def draw_pipe(self, window):
        self.top_rect = pygame.Rect(self.x, 0, Pipe.width, self.top_part_length)
        self.bottom_rect = pygame.Rect(self.x, Ground.ground_level - self.bottom_part_length, Pipe.width, self.bottom_part_length)

        pygame.draw.rect(window, self.color, self.top_rect)
        pygame.draw.rect(window, self.color, self.bottom_rect)

    def update(self):
        self.x -= 1

        if self.x + Pipe.width < 0:
            self.offscreen = True

class Bird:
    size = 20
    init_x_loc = 50
    init_y_loc = 200
    vel_incr = 0.25
    vel_cap = 5

    def __init__(self, net):
        self.x, self.y = self.init_x_loc, self.init_y_loc
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        
        # random colors for multiple generations
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        # physics
        self.vel = 0
        self.is_flapping = False
        self.alive = True
        self.lifespan = 0

        # ai needed stuff
        self.pipes_passed = 0 # used for bonus rewards
        self.punishment = 0 # used if bird is stupid and just hangs out up top
        self.vision = [0.5, 1, 0.5]
        self.fitness = 0
        self.net = net

    def draw_bird(self):
        pygame.draw.rect(window, self.color, self.rect)

    def test_ground_collision(self, ground_rect):
        return pygame.Rect.colliderect(self.rect, ground_rect)
    
    def test_pipe_collision(self):
        for pipe in pipes:
            if pygame.Rect.colliderect(self.rect, pipe.top_rect):
                return True
            if pygame.Rect.colliderect(self.rect, pipe.bottom_rect):
                return True
        return False

    def test_sky_collision(self):
        return self.rect.y < 0
    
    def check_pipe_passed(self):
        closest_pipe = self.closest_pipe()

        # check if there is a closest pipe on screen, check if bird is to its right, check if pipe not already passed
        if closest_pipe and self.rect.left > closest_pipe.x + closest_pipe.width and not closest_pipe.passed:
            closest_pipe.passed = True
            self.pipes_passed += 1

    def update(self, ground_rect):
        self.check_pipe_passed()
        if self.test_ground_collision(ground_rect) or self.test_pipe_collision():
            self.alive = False
            self.is_flapping = False
            self.vel = 0
        else:
            # Increment velocity
            self.vel += Bird.vel_incr
            # Cap the velocity
            if self.vel > Bird.vel_cap:
                self.vel = Bird.vel_cap
            # Apply velocity to the y position
            self.rect.y += int(self.vel)
            self.lifespan += 1

    def flap(self):
        if not self.is_flapping and not self.test_sky_collision():
            self.is_flapping = True
            self.vel = -5
        elif self.test_sky_collision():
            self.punishment += 2 # punish for staying too high up
        if self.vel >= 1: # ensure bird begins falling down again slightly before flapping again
            self.is_flapping = False

    def closest_pipe(self):
        min_dist = WIN_WIDTH + 1
        for pipe in pipes:
            dist = pipe.x - self.x
            if dist > -self.size and not pipe.passed and dist < min_dist:
                min_dist = dist
                closest_pipe = pipe
            
        for pipe in pipes:
            if pipe == closest_pipe:
                pipe.color = (200, 0, 0)
            else:
                pipe.color = (220,220,220)
        
        return closest_pipe
            
    def look(self):
        if pipes:
            # line to top pipe
            player_y = self.rect.center[1]
            player_x = self.rect.center[0]
            bottom_of_top_pipe = self.closest_pipe().top_rect.bottom
            left_of_closest_pipe = self.closest_pipe().x
            top_of_bottom_pipe = self.closest_pipe().bottom_rect.top

            # line of sight from bird to bottom edge of closest top pipe
            self.vision[0] = max(0, player_y - bottom_of_top_pipe) / Ground.ground_level
            # line of sight from bird to left edge of closest pipe
            self.vision[1] = max(0, left_of_closest_pipe - player_x) / WIN_WIDTH
            # line of sight from bird to top edge of closest bottom pipe
            self.vision[2] = max(0, top_of_bottom_pipe - player_y) / Ground.ground_level
            # draw the vision lines
            pygame.draw.line(window, self.color, self.rect.center, (player_x, bottom_of_top_pipe))
            pygame.draw.line(window, self.color, self.rect.center, (left_of_closest_pipe, player_y))
            pygame.draw.line(window, self.color, self.rect.center, (player_x, top_of_bottom_pipe))

    def evaluate_fitness(self):
        # fitness is how many frames bird is alive + bonus for passing pipes
        self.fitness = self.lifespan + (10 * self.pipes_passed) - self.punishment

        if not self.alive and self.pipes_passed == 0:
            self.fitness *= 0.5

        self.fitness = max(0, self.fitness)

    def think(self):
        self.look()
        output = self.net.feed_forward(self.vision)
        if output[0] > 0.5:
            self.flap()

ground = Ground()
#bird = Bird()