import pygame
import sys
import turtle
import time

class Game:
    RED = (255, 20, 60)
    BLUE = (0, 200, 255)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    def __init__(self, rx, ry, bx, by):
        with open("winners.txt", 'w') as file:
            pass
        pygame.init()
        self.WIDTH = 1024
        self.HEIGHT = 720
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Cycle Clash")

        self.clock = pygame.time.Clock()
        
        self.running = True

        self.red_head = [float(rx), float(ry)]
        self.blue_head = [float(bx), float(by)]
        self.red_trail = [self.red_head[:]]
        self.blue_trail = [self.blue_head[:]]

        self.red_dir = [1, 0]
        self.blue_dir = [-1, 0]

        self.trail_thickness = 5
        self.speed = 5

        self.red_score = 0
        self.blue_score = 0
        self.game_over = False
        self.winner = ""

    def play_startup_animation(self):
        try:
            screen = turtle.Screen()
            screen.bgcolor("black")
            screen.title("Red and Blue Dot Animation")

            red_dot = turtle.Turtle()
            blue_dot = turtle.Turtle()

            # Hide the turtle pointers
            red_dot.hideturtle()
            blue_dot.hideturtle()

            red_dot.color("red")
            red_dot.penup()

            blue_dot.color("blue")
            blue_dot.penup()

            # Draw expanding dots
            for size in range(1, 100, 2):  # Increment size
                red_dot.goto(0, 0)
                blue_dot.goto(0, 0)
                red_dot.dot(size)
                blue_dot.dot(size - 10)

                # Pause for animation effect
                time.sleep(0.05)

            # Pause for 3 seconds before closing the window
            time.sleep(3)
            screen.bye()
        except Exception as e:
            print(f"Startup animation failed: {e}")

    def display_rules(self):
        """Display game rules before starting."""
        try:
            screen = turtle.Screen()
            screen.bgcolor("black")
            screen.title("Game Rules")

            rule_turtle = turtle.Turtle()
            rule_turtle.hideturtle()
            rule_turtle.color("white")
            rule_turtle.penup()
            rule_turtle.goto(0, 200)

            rules = [
                "1. Use WASD for Red Player to control direction.",
                "2. Use Arrow Keys for Blue Player to control direction.",
                "3. Avoid colliding with walls or trails.",
                "4. The first player to collide loses.",
                "Press SPACE to start the game."
            ]

            for rule in rules:
                rule_turtle.write(rule, align="center", font=("Arial", 20, "normal"))
                rule_turtle.goto(0, rule_turtle.ycor() - 40)  # Move down after each rule

            time.sleep(7)

            screen.bye()
        except Exception as e:
            print(f"Failed to display rules: {e}")

    def draw_trail(self, trail, color):
        for i in range(1, len(trail)):
            start = trail[i - 1]
            end = trail[i]
            pygame.draw.line(self.screen, color, start, end, self.trail_thickness)

    def update_position(self):
        self.red_head[0] += self.red_dir[0] * self.speed
        self.red_head[1] += self.red_dir[1] * self.speed
        self.red_trail.append(self.red_head[:])

        self.blue_head[0] += self.blue_dir[0] * self.speed
        self.blue_head[1] += self.blue_dir[1] * self.speed
        self.blue_trail.append(self.blue_head[:])

    def check_collision(self, player_head, trail):
        if player_head[0] < 0 or player_head[0] >= self.WIDTH or player_head[1] < 0 or player_head[1] >= self.HEIGHT:
            return True

        for segment in trail:
            if abs(player_head[0] - segment[0]) < self.trail_thickness and abs(player_head[1] - segment[1]) < self.trail_thickness:
                return True
        return False

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and self.red_dir != [0, 1]:
            self.red_dir = [0, -1]
        elif keys[pygame.K_s] and self.red_dir != [0, -1]:
            self.red_dir = [0, 1]
        elif keys[pygame.K_a] and self.red_dir != [1, 0]:
            self.red_dir = [-1, 0]
        elif keys[pygame.K_d] and self.red_dir != [-1, 0]:
            self.red_dir = [1, 0]

        if keys[pygame.K_UP] and self.blue_dir != [0, 1]:
            self.blue_dir = [0, -1]
        elif keys[pygame.K_DOWN] and self.blue_dir != [0, -1]:
            self.blue_dir = [0, 1]
        elif keys[pygame.K_LEFT] and self.blue_dir != [1, 0]:
            self.blue_dir = [-1, 0]
        elif keys[pygame.K_RIGHT] and self.blue_dir != [-1, 0]:
            self.blue_dir = [1, 0]

    def display_game_over(self):
        font = pygame.font.Font(None, 74)
        game_over_text = font.render(f"{self.winner} Wins!", True, self.WHITE)
        button_font = pygame.font.Font(None, 50)
        button_text = button_font.render("Play Again", True, self.BLACK)
        button_rect = pygame.Rect(self.WIDTH // 2 - 100, self.HEIGHT // 2 + 50, 200, 60)

        pygame.draw.rect(self.screen, self.WHITE, button_rect)
        self.screen.blit(game_over_text, (self.WIDTH // 2 - game_over_text.get_width() // 2, self.HEIGHT // 2 - 100))
        self.screen.blit(button_text, (button_rect.x + 20, button_rect.y + 10))
        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        waiting = False
                        self.reset_game()
        

    def reset_game(self):
        self.red_head = [200, 400]
        self.blue_head = [800, 400]
        self.red_trail = [self.red_head[:]]
        self.blue_trail = [self.blue_head[:]]
        self.red_dir = [1, 0]
        self.blue_dir = [-1, 0]
        self.game_over = False
        self.winner = ""

    def run(self):
        try:
            self.play_startup_animation()
        except:
            print("Failed to play startup animation.")
        
        self.display_rules()

        while self.running:
            self.screen.fill(self.BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if self.game_over:
                if self.winner == "Red":
                    self.screen.fill(self.RED)
                elif self.winner == "Blue":
                    self.screen.fill(self.BLUE)
                pygame.display.update()
                pygame.time.delay(2000) 

                self.display_game_over()

                with open("winners.txt", "a") as win:
                    win.write(f"Winner: {self.winner}\n")
            else:
                self.handle_input()
                self.update_position()

                if self.check_collision(self.red_head, self.blue_trail) or self.check_collision(self.red_head, self.red_trail[:-1]):
                    self.game_over = True
                    self.winner = "Blue"
                    with open("winners.txt", "a") as win:
                        win.write("Winner: Blue\n")
                if self.check_collision(self.blue_head, self.red_trail) or self.check_collision(self.blue_head, self.blue_trail[:-1]):
                    self.game_over = True
                    self.winner = "Red"
                    with open("winners.txt", "a") as win:
                        win.write("Winner: Red\n")

                self.draw_trail(self.red_trail, self.RED)
                self.draw_trail(self.blue_trail, self.BLUE)

                pygame.draw.circle(self.screen, self.RED, (int(self.red_head[0]), int(self.red_head[1])), self.trail_thickness)
                pygame.draw.circle(self.screen, self.BLUE, (int(self.blue_head[0]), int(self.blue_head[1])), self.trail_thickness)

            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    game = Game(200, 400, 800, 400)
    game.run()
