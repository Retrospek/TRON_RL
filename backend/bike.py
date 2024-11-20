import pygame

class Bike:
    def __init__(self, color, x, y, screen, width, height):
        self.color = color
        self.x = x
        self.y = y
        self.direction = (0, 1)  # Initial direction is moving down
        self.trail = [(self.x, self.y)]
        self.alive = True
        self.speed = 10  # Increased speed (moving more pixels per frame)
        self.screen = screen
        self.width = width
        self.height = height

    def move(self):
        # Move the bike by adding the direction to the position
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed
        self.trail.append((self.x, self.y))

    def change_direction(self, direction):
        # Prevent turning 180 degrees directly (can't go back on itself)
        if (self.direction[0] == -direction[0] and self.direction[1] == -direction[1]):
            return  # Don't allow 180-degree turns
        self.direction = direction

    def draw(self):
        for segment in self.trail:
            pygame.draw.rect(self.screen, self.color, pygame.Rect(segment[0], segment[1], 10, 10))

    def check_collision(self):
        # Collision with walls
        if self.x < 0 or self.x >= self.width or self.y < 0 or self.y >= self.height:
            self.alive = False  # Collision with wall
        # Collision with own trail
        if (self.x, self.y) in self.trail[:-1]:
            self.alive = False  # Collision with own trail
