import src.components as comps

class Population:
    def __init__(self, size):
        self.size = size
        self.birds = [comps.Bird() for _ in range(0, self.size)]

    def update_live_players(self):
        for bird in self.birds:
            if bird.alive:
                bird.look()
                bird.think()
                bird.draw_bird(comps.window)
                bird.update(comps.ground)

    def is_extinct(self):
        extinct = True
        for bird in self.birds:
            if bird.alive:
                extinct = False
                break
        
        return extinct