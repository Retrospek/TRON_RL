import pygame
import random
from bike import Bike

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tron Lightcycle Game")

WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)

FPS = 30
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 30)

def show_text(text, x, y, font, color):
    label = font.render(text, True, color)
    screen.blit(label, (x, y))

def game_start_screen():
    screen.fill(BLACK)
    show_text("Tron Lightcycle Game", WIDTH // 2 - 150, HEIGHT // 4, font, WHITE)
    show_text("Control the bikes to avoid crashing!", WIDTH // 2 - 150, HEIGHT // 3, font, WHITE)
    show_text("Press Arrow keys to control the bikes.", WIDTH // 2 - 180, HEIGHT // 2, font, WHITE)
    show_text("Left/Right arrows to turn, Up/Down to move.", WIDTH // 2 - 180, HEIGHT // 1.8, font, WHITE)
    show_text("Game starts in:", WIDTH // 2 - 100, HEIGHT // 1.5, font, WHITE)
    
    pygame.display.update()
    pygame.time.wait(1000)
    
    for countdown in range(5, 0, -1):
        screen.fill(BLACK)
        show_text(f"Game starts in: {countdown}", WIDTH // 2 - 100, HEIGHT // 2, font, WHITE)
        pygame.display.update()
        pygame.time.wait(1000)
    
    pygame.time.wait(500)

class GAME_INFO:
    def __init__(self, bikeM_trail, bikeC_trail, bikeM_Score, bikeC_Score):
        self.bikeM_trail = bikeM_trail
        self.bikeC_trail = bikeC_trail
        self.bikeM_Score = bikeM_Score
        self.bikeC_Score = bikeC_Score

class Game:

    WHITE = (255, 255, 255)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    BLACK = (0, 0, 0)
    FONT = pygame.font.SysFont("comicsans", 50)

    def __init__(self, window):
        self.WIDTH, self.HEIGHT = 800, 600
        self.window = window
        self.bike1 = Bike(self.CYAN, self.WIDTH // 4, self.HEIGHT // 2)
        self.bike2 = Bike(self.MAGENTA, 3 * self.WIDTH // 4, self.HEIGHT // 2)
        self.Bike_M_Score = 0
        self.Bike_C_Score = 0
        self.Bik_M_Trail = 0
        self.Bik_C_Trail = 0
        self.Bike_M_Trajectory = []
        self.Bike_C_Trajectory = []

    def show_text(self, text, x, y, color):
        label = self.FONT.render(text, True, color)
        self.window.blit(label, (x, y))

    def game_start_screen(self):
        self.window.fill(self.BLACK)
        self.show_text("Tron Lightcycle Game", self.WIDTH // 2 - 150, self.HEIGHT // 4, self.WHITE)
        self.show_text("Control the bikes to avoid crashing!", self.WIDTH // 2 - 150, self.HEIGHT // 3, self.WHITE)
        self.show_text("Press Arrow keys to control the bikes.", self.WIDTH // 2 - 180, self.HEIGHT // 2, self.WHITE)
        self.show_text("Left/Right arrows to turn, Up/Down to move.", self.WIDTH // 2 - 180, self.HEIGHT // 1.8, self.WHITE)
        self.show_text("Game starts in:", self.WIDTH // 2 - 100, self.HEIGHT // 1.5, self.WHITE)
        pygame.display.update()
        pygame.time.wait(1000)
        
        for countdown in range(5, 0, -1):
            self.window.fill(self.BLACK)
            self.show_text(f"Game starts in: {countdown}", self.WIDTH // 2 - 100, self.HEIGHT // 2, self.WHITE)
            pygame.display.update()
            pygame.time.wait(1000)
        
        pygame.time.wait(500)

    def main_game_loop(self):
        while self.bike1.alive and self.bike2.alive:
            self.window.fill(self.BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                self.bike1.change_direction((-1, 0))
            elif keys[pygame.K_RIGHT]:
                self.bike1.change_direction((1, 0))
            elif keys[pygame.K_UP]:
                self.bike1.change_direction((0, -1))
            elif keys[pygame.K_DOWN]:
                self.bike1.change_direction((0, 1))

            if keys[pygame.K_a]:
                self.bike2.change_direction((-1, 0))
            elif keys[pygame.K_d]:
                self.bike2.change_direction((1, 0))
            elif keys[pygame.K_w]:
                self.bike2.change_direction((0, -1))
            elif keys[pygame.K_s]:
                self.bike2.change_direction((0, 1))

            self.bike1.move()
            self.bike2.move()

            self.bike1.check_collision()
            self.bike2.check_collision()

            self.bike1.draw(self.window)
            self.bike2.draw(self.window)

            pygame.display.update()
            clock.tick(FPS)

        self.window.fill(self.BLACK)
        if not self.bike1.alive:
            self.show_text("Game Over! Bike 2 Wins!", self.WIDTH // 2 - 150, self.HEIGHT // 2, self.WHITE)
        elif not self.bike2.alive:
            self.show_text("Game Over! Bike 1 Wins!", self.WIDTH // 2 - 150, self.HEIGHT // 2, self.WHITE)

        pygame.display.update()
