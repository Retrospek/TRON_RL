import pygame
import random

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tron Lightcycle Game")

# Colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)

# FPS: Increase it for faster movement
FPS = 30  # Increased FPS to make the game run faster
clock = pygame.time.Clock()

# Font for text
font = pygame.font.SysFont("Arial", 30)

# Bike class
class Bike:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y
        self.direction = (0, 1)  # Initial direction is moving down
        self.trail = [(self.x, self.y)]
        self.alive = True
        self.speed = 10  # Increased speed (moving more pixels per frame)

    def move(self):
        # Move the bike by adding the direction to the position
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed
        self.trail.append((self.x, self.y))

    def change_direction(self, direction):
        self.direction = direction

    def draw(self):
        for segment in self.trail:
            pygame.draw.rect(screen, self.color, pygame.Rect(segment[0], segment[1], 10, 10))

    def check_collision(self):
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            self.alive = False  # Collision with wall
        if (self.x, self.y) in self.trail[:-1]:
            self.alive = False  # Collision with own trail


def show_text(text, x, y, font, color):
    """Helper function to display text on screen."""
    label = font.render(text, True, color)
    screen.blit(label, (x, y))

def game_start_screen():
    """Show instructions and countdown before the game starts."""
    screen.fill(BLACK)
    
    # Display the instructions
    show_text("Tron Lightcycle Game", WIDTH // 2 - 150, HEIGHT // 4, font, WHITE)
    show_text("Control the bikes to avoid crashing!", WIDTH // 2 - 150, HEIGHT // 3, font, WHITE)
    show_text("Press Arrow keys to control the bikes.", WIDTH // 2 - 180, HEIGHT // 2, font, WHITE)
    show_text("Left/Right arrows to turn, Up to go forward.", WIDTH // 2 - 180, HEIGHT // 1.8, font, WHITE)
    show_text("Game starts in:", WIDTH // 2 - 100, HEIGHT // 1.5, font, WHITE)
    
    pygame.display.update()
    pygame.time.wait(1000)  # Wait for 1 second to show instructions
    
    # Countdown from 10 to 1
    for countdown in range(5, 0, -1):
        screen.fill(BLACK)
        show_text(f"Game starts in: {countdown}", WIDTH // 2 - 100, HEIGHT // 2, font, WHITE)
        pygame.display.update()
        pygame.time.wait(1000)  # Wait for 1 second per countdown
    
    pygame.time.wait(500)  # Wait for a moment after countdown ends


def main_game_loop():
    """Main game loop."""
    # Initialize bikes
    bike1 = Bike(CYAN, WIDTH // 4, HEIGHT // 2)
    bike2 = Bike(MAGENTA, 3 * WIDTH // 4, HEIGHT // 2)
    
    # Game loop
    while bike1.alive and bike2.alive:
        screen.fill(BLACK)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Get the keys pressed for controlling the bikes
        keys = pygame.key.get_pressed()

        # Control bike 1
        if keys[pygame.K_LEFT]:
            bike1.change_direction((-1, 0))  # Move left
        elif keys[pygame.K_RIGHT]:
            bike1.change_direction((1, 0))  # Move right
        elif keys[pygame.K_UP]:
            bike1.change_direction((0, -1))  # Move up
        
        # Control bike 2
        if keys[pygame.K_a]:
            bike2.change_direction((-1, 0))  # Move left
        elif keys[pygame.K_d]:
            bike2.change_direction((1, 0))  # Move right
        elif keys[pygame.K_w]:
            bike2.change_direction((0, -1))  # Move up

        # Move both bikes
        bike1.move()
        bike2.move()

        # Check for collisions
        bike1.check_collision()
        bike2.check_collision()

        # Draw the bikes
        bike1.draw()
        bike2.draw()

        # Update the screen
        pygame.display.update()

        # Set the FPS
        clock.tick(FPS)

    # Display game over message
    screen.fill(BLACK)
    if not bike1.alive:
        show_text("Game Over! Bike 2 Wins!", WIDTH // 2 - 150, HEIGHT // 2, font, WHITE)
    elif not bike2.alive:
        show_text("Game Over! Bike 1 Wins!", WIDTH // 2 - 150, HEIGHT // 2, font, WHITE)
    
    pygame.display.update()
    #pygame.time.wait(2000)  # Wait for 2 seconds before closing the game


if __name__ == "__main__":
    # Show the start screen with instructions and countdown
    game_start_screen()
    
    # Start the main game loop after countdown
    main_game_loop()

    # Quit pygame
    pygame.quit()
