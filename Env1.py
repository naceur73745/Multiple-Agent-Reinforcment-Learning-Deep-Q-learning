import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy  as np
import os 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time  
import numpy as np 
import pandas as pd 
from MultiAgent import Agent 
import math


# Import the Pygame library and initialize it
pygame.init()

# Set the font and size for text rendering
font = pygame.font.SysFont("comicsans", 50)
# Alternatively, you can use Arial font with a size of 25
# font = pygame.font.SysFont('arial', 25)

# Define an enumeration for directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define a named tuple for representing points with 'x' and 'y' attributes
Point = namedtuple('Point', 'x, y')

# Define color constants
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Set the size of each block in the game grid
BLOCK_SIZE = 30

# Set the speed of the game
SPEED = 50

# Set the frames per second for the game
FPS = 50

# Specify the number of apples initially present in the game
NUM_APPLES = 2


class SnakeGame:

    def __init__(self, w=600, h=600, num_snakes=1):
        """
        Initializes the game environment.

        Parameters:
            w (int): Width of the game window.
            h (int): Height of the game window.
            num_snakes (int): Number of snakes in the game.

        Attributes:
            w (int): Width of the game window.
            h (int): Height of the game window.
            background_image (pygame.Surface): Background image of the game window.
            display (pygame.Surface): Pygame display window.
            clock (pygame.time.Clock): Pygame clock for controlling the frame rate.
            num_snakes (int): Number of snakes in the game.
            snakes (list): List to store snake instances.
            frames_since_last_action (list): List to keep track of frames since the last action for each snake.
            MAX_FRAMES_INACTIVITY (int): Maximum frames allowed for inactivity.
            start_time (list): List to store the start time for each snake.
            snake_colors (list): List of predefined colors for snakes.
        """
        # Set the width and height of the game window
        self.w = w
        self.h = h

        # Load and scale the background image
        self.background_image = pygame.transform.scale(pygame.image.load("Brain.jpg"), (self.w, self.h))

        # Create the game display window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

        # Initialize the game clock
        self.clock = pygame.time.Clock()

        # Set the number of snakes in the game
        self.num_snakes = num_snakes

        # Initialize lists to store snake instances, frames since last action, and start time
        self.snakes = []
        self.frames_since_last_action = [0] * self.num_snakes
        self.start_time = []

        # Maximum frames allowed for inactivity
        self.MAX_FRAMES_INACTIVITY = 1000

        # Predefined colors for snakes
        self.snake_colors = [
            "white",
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
            "orange",
            "gray"
        ]

        # Call the reset method to initialize/reset the game state
        self.reset()


    def reset_snake(self, snake_index):
        """
        Resets the state of a specific snake in the game.

        Parameters:
            snake_index (int): Index of the snake to be reset.

        Resets the snake's position, direction, score, game over status, start time, and the count of eaten apples.
        """
        # Generate random starting coordinates for the snake's head within the game window
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        # Set the initial state for the specified snake
        self.heads[snake_index] = Point(x, y)
        self.snakes[snake_index] = [
            self.heads[snake_index],
            Point(self.heads[snake_index].x - BLOCK_SIZE, self.heads[snake_index].y),
            Point(self.heads[snake_index].x - (2 * BLOCK_SIZE), self.heads[snake_index].y)
        ]
        self.directions[snake_index] = Direction.RIGHT
        self.score[snake_index] = 0
        self.game_over[snake_index] = False
        self.start_time[snake_index] = time.time()
        self.Apple_EatenSnakes[snake_index] = [0, 0]


    def reset(self):
        """
        Resets the entire game state.

        Initializes or resets various attributes, including frame count, start time, game over status,
        score, eaten apples count, directions, snake heads, snake bodies, and initial food placement.

        This method is called to set up or reset the game environment.
        """
        # Reset the frame iteration count
        self.frame_iteration = 0

        # Initialize start time for each snake
        for _ in range(self.num_snakes):
            self.start_time.append(time.time())

        # Set game over status, score, and eaten apples count for each snake
        self.game_over = [False] * (self.num_snakes)
        self.score = [0] * (self.num_snakes)
        self.Apple_EatenSnakes = [[0, 0]] * (self.num_snakes)

        # Set initial direction for each snake
        self.directions = [Direction.RIGHT for _ in range(self.num_snakes)]

        # Initialize lists to store snake head positions and bodies
        self.heads = []  # List to store snake head positions
        self.snakes = []  # List to store snake body segments

        # Generate random starting positions for each snake
        for _ in range(self.num_snakes):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            head = Point(x, y)
            self.heads.append(head)
            snake = [head, Point(head.x - BLOCK_SIZE, head.y), Point(head.x - (2 * BLOCK_SIZE), head.y)]
            self.snakes.append(snake)

        # Initialize the list to store apple positions
        self.food = []
        
        # Generate initial apples on the game grid
        self._place_food()

 
    def get_state(self):
        """
        Retrieves the current game state for each snake.

        Returns:
            list: List of numpy arrays representing the state of each snake.

        The state includes information about the snake's surroundings, danger zones, move direction,
        positions and lengths of other snakes, actions of other snakes, and the location of food.
        """
        states = []

        # Iterate over each snake in the game
        for snake_index in range(self.num_snakes):

            # Access the head of the snake
            head = self.heads[snake_index]

            # Define points in the vicinity of the snake
            point_l = Point(head.x - BLOCK_SIZE, head.y)
            point_r = Point(head.x + BLOCK_SIZE, head.y)
            point_u = Point(head.x, head.y - BLOCK_SIZE)
            point_d = Point(head.x, head.y + BLOCK_SIZE)

            # Check the direction of the snake
            dir_l = self.directions[snake_index] == Direction.LEFT
            dir_r = self.directions[snake_index] == Direction.RIGHT
            dir_u = self.directions[snake_index] == Direction.UP
            dir_d = self.directions[snake_index] == Direction.DOWN

            # Collect positions, lengths, and actions of other snakes, and color of other snakes
            opponent_positions = []
            opponent_lengths = []
            opponent_actions = []

            for snake_idx in range(self.num_snakes):
                if snake_idx != snake_index:
                    opponent_positions.append(self.heads[snake_idx])
                    opponent_lengths.append(len(self.snakes[snake_idx]))

                    # Get the opponent's action
                    opponent_action = [0, 0, 0, 0]
                    if self.directions[snake_idx] == Direction.LEFT:
                        opponent_action = [1, 0, 0, 0]
                    elif self.directions[snake_idx] == Direction.RIGHT:
                        opponent_action = [0, 1, 0, 0]
                    elif self.directions[snake_idx] == Direction.UP:
                        opponent_action = [0, 0, 1, 0]
                    elif self.directions[snake_idx] == Direction.DOWN:
                        opponent_action = [0, 0, 0, 1]
                    opponent_actions.append(opponent_action)

            # Define the state based on the snake's surroundings and interactions with other snakes
            state = [
                int((dir_r and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_d)))),
                # Danger right
                int((dir_u and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_d)))),
                # Danger left
                int((dir_d and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_d)))),

                # Move direction
                int(dir_l),
                int(dir_r),
                int(dir_u),
                int(dir_d)
            ]

            # Add the opponent's move
            for action in opponent_actions:
                state += action

            # Add lengths comparison between the current snake and other snakes
            my_length = len(self.snakes[snake_index])
            for opponent_length in opponent_lengths:
                if my_length > opponent_length:
                    state += [1, 0]  # Snake length is greater than opponent
                elif my_length < opponent_length:
                    state += [0, 1]  # Snake length is smaller than opponent
                else:
                    state += [0, 0]  # Snake length is equal to opponent

            # Add the food location relative to the snake's head
            for food_item in self.food:
                state += [
                    int(food_item.x < head.x),  # Food left
                    int(food_item.x > head.x),  # Food right
                    int(food_item.y < head.y),  # Food up
                    int(food_item.y > head.y)   # Food down
                ]

            # Append the state as a numpy array to the list of states
            states.append(np.array(state, dtype=int))

        return states




    
    def CheckForGetState(self, snake_index, pt=None):
        """
        Checks for collisions or boundary hits for a specific snake at a given point.

        Parameters:
            snake_index (int): Index of the snake to check.
            pt (Point, optional): Point to check for collisions. Defaults to None, in which case the head of the snake is used.

        Returns:
            bool: True if there is a collision or boundary hit, False otherwise.

        This method checks if the given point (or the head of the snake if not provided) hits the game boundaries,
        collides with the snake itself, or collides with other snakes. Returns True if any collision is detected, False otherwise.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Check if the point hits the snake itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

        # Check if the point hits other snakes
        for i, snake in enumerate(self.snakes):
            if i != snake_index:
                if pt in [body for body in snake]:
                    return True

        # No collision or boundary hit detected
        return False

    

    def _place_food(self):
        """
        Places food items on the game grid.

        Randomly generates and places a specified number of food items on the game grid.

        This method is called during the initialization of the game and when a snake consumes an apple.
        """
        self.food = []
        for _ in range(NUM_APPLES):  # NUM_APPLES is the desired number of apples
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food.append(Point(x, y))


    def calculate_distance(self, point1, point2):
        """
        Calculates the Euclidean distance between two points.

        Parameters:
            point1 (Point): First point.
            point2 (Point): Second point.

        Returns:
            float: Euclidean distance between the two points.

        This method calculates and returns the Euclidean distance between two points in the game grid.
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    
 
    def play_step(self, actions):
        """
        Performs a single step of the game based on the given actions.

        Parameters:
            actions (list): List of actions corresponding to each snake.

        Returns:
            tuple: A tuple containing reward, game over status, scores, snake information, total time played,
            apples eaten by snakes, distances to other snakes, and distances to apples.

        This method updates the game state based on the actions of the snakes and returns various game-related information.
        """
        self.frame_iteration += 1

        # Handle pygame events, such as quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialize lists to store various information for each snake
        snake_info = [[] for _ in range(self.num_snakes)]
        total_time_played = [[] for _ in range(self.num_snakes)]
        rewards = [0] * self.num_snakes

        # Move the snakes based on the given actions
        self._move(actions)

        # Lists to store distances to apples and other snakes for each snake
        AppleDistanceList = []
        DistanceToSnakesList = []

        # Iterate over each snake in the game
        for snake_index in range(self.num_snakes):
            AppleDistance = []
            DistanceToSnakes = []

            # Calculate distances to each food item (apple) for the current snake
            for i, food in enumerate(self.food):
                apple_distance = self.calculate_distance(self.snakes[snake_index][0], self.food[i])
                AppleDistance.append(apple_distance)

            AppleDistanceList.append(AppleDistance)

            # Calculate distances to each snake (including itself) for the current snake
            for i, snake in enumerate(self.snakes):
                if i != snake_index:
                    snake_distance = self.calculate_distance(self.snakes[snake_index][0], self.snakes[i][0])
                    DistanceToSnakes.append(snake_distance)

            DistanceToSnakesList.append(DistanceToSnakes)

            elapsed_time = time.time() - self.start_time[snake_index]
            snake_head = self.heads[snake_index]

            # Check for collisions with the wall
            if self.collsion_wall(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = -60
                self.frames_since_last_action[snake_index] = 0
                snake_info[snake_index].append("I collided with the wall")

            # Check for collisions with itself
            elif self.collison_with_itself(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = -50
                self.frames_since_last_action[snake_index] = 0
                snake_info[snake_index].append("I collided with myself")

            else:
                # Move the snake's head and update the game state
                self.snakes[snake_index].insert(0, snake_head)

                # Check if the snake ate an apple
                if snake_head in self.food:
                    self.score[snake_index] += 1
                    self.Apple_EatenSnakes[snake_index][0] += 1

                    # Determine the reward for eating the apple
                    for i, apple in enumerate(self.food):
                        if snake_head == apple:
                            rewards[snake_index] = 76
                            self.food.pop(i)  # Remove the eaten apple
                            self._place_food()  # Generate a new apple to replace the eaten one
                            break

                    self.frames_since_last_action[snake_index] = 0
                    snake_info[snake_index].append("I ate an apple, yum!")

                else:
                    # If the snake did not eat an apple, remove the last segment of its tail
                    self.snakes[snake_index].pop()
                    self.frames_since_last_action[snake_index] += 1

                    # Encourage collisions with other snakes
                    for other_snake_index in range(self.num_snakes):
                        if snake_index != other_snake_index:
                            if snake_head in self.snakes[other_snake_index]:
                                if len(self.snakes[snake_index]) <= len(self.snakes[other_snake_index]):
                                    rewards[snake_index] = -70  # Encourage collision with longer snakes
                                    snake_info[snake_index].append("Another snake ate me!")
                                    self.game_over[snake_index] = True

                                elif len(self.snakes[snake_index]) > len(self.snakes[other_snake_index]):
                                    rewards[snake_index] = 70  # Encourage collision with shorter snakes
                                    snake_info[snake_index].append("I ate a snake!")
                                    self.Apple_EatenSnakes[snake_index][1] += 1
                                    self.score[snake_index] += 1
                                else:
                                    snake_info[snake_index].append("Exploring the environment!")

                    total_time_played[snake_index] = elapsed_time

                    # Check if nothing has happened for a long time
                    if self.frames_since_last_action[snake_index] >= self.MAX_FRAMES_INACTIVITY:
                        self.game_over[snake_index] = True
                        rewards[snake_index] = -10
                        self.reset_snake(snake_index)
                        self.frames_since_last_action[snake_index] = 0
                        snake_info[snake_index].append("I didn't do anything for n iterations")

        # Return various game-related information as a tuple
        return rewards, self.game_over, self.score, snake_info, total_time_played, self.Apple_EatenSnakes, DistanceToSnakesList, AppleDistanceList
    

    def is_collision(self, snake_index, pt=None):
        """
        Checks for collisions at a given point for a specific snake.

        Parameters:
            snake_index (int): Index of the snake to check.
            pt (Point, optional): Point to check for collisions. Defaults to None, in which case the head of the snake is used.

        Returns:
            bool: True if there is a collision, False otherwise.

        This method checks if the given point (or the head of the snake if not provided) hits the game boundaries or collides with itself.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Check if the point hits the snake itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

        # No collision detected
        return False


    def collsion_wall(self, snake_index, pt=None):
        """
        Checks for collisions with the game wall at a given point for a specific snake.

        Parameters:
            snake_index (int): Index of the snake to check.
            pt (Point, optional): Point to check for collisions. Defaults to None, in which case the head of the snake is used.

        Returns:
            bool: True if there is a collision with the wall, False otherwise.

        This method checks if the given point (or the head of the snake if not provided) hits the game boundaries.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # No collision with the wall detected
        return False


    def collison_with_itself(self, snake_index, pt=None):
        """
        Checks for collisions with itself at a given point for a specific snake.

        Parameters:
            snake_index (int): Index of the snake to check.
            pt (Point, optional): Point to check for collisions. Defaults to None, in which case the head of the snake is used.

        Returns:
            bool: True if there is a collision with itself, False otherwise.

        This method checks if the given point (or the head of the snake if not provided) collides with the snake itself.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

        # No collision with itself detected
        return False



    def eate_other_snake(self, snake_index):
        """
        Checks if the head of a snake has collided and eaten another snake.

        Parameters:
            snake_index (int): Index of the snake to check.

        Returns:
            tuple: A tuple containing a boolean indicating whether the snake ate another snake,
                and a list of indices of the collided snakes.

        This method checks if the head of the given snake has collided with and eaten another snake.
        It returns a tuple containing a boolean value indicating the result and a list of indices of the collided snakes.
        """
        head = self.heads[snake_index]
        truth = False
        collided = []

        # Iterate over all snakes in the game
        for i, snake in enumerate(self.snakes):
            # Check if the snake is not the current snake and if its head is at the same position as the current snake's head
            if i != snake_index and head in snake:
                # Get the index of the collided snake
                collided.append(i)
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])

                # Check if the current snake is longer than the collided snake
                if current_snake_length > collided_snake_length:
                    truth = True

        return truth, collided

    
    def eaten_by_other_snake(self, snake_index):
        """
        Checks if the snake has been eaten by another snake.

        Parameters:
            snake_index (int): Index of the snake to check.

        Returns:
            bool: True if the snake has been eaten by another snake, False otherwise.

        This method checks if the head of the given snake has been collided with and eaten by another snake.
        It returns a boolean value indicating the result.
        """
        head = self.heads[snake_index]

        # Iterate over all snakes in the game
        for i, snake in enumerate(self.snakes):
            # Check if the snake is not the current snake and if its head is at the same position as the current snake's head
            if i != snake_index and head in snake:
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])

                # Check if the current snake is shorter or equal to the collided snake
                if current_snake_length <= collided_snake_length:
                    return True

        # The snake has not been eaten by any other snake
        return False

    

    def grid(self):
        """
        Draws a grid on the game display.

        This method draws a grid on the game display by iterating over rows and columns and drawing rectangles with a specified block size.
        The drawn grid is updated on the game display.

        Note: It is assumed that the method is called within a Pygame loop to update the display continuously.
        """
        for row in range(0, self.h, BLOCK_SIZE):
            for col in range(0, self.h, BLOCK_SIZE):
                # Draw a rectangle
                rect = pygame.Rect(row, col, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, "green", rect, 3)

        pygame.display.update()


    def _update_ui(self):
        """
        Updates the game user interface (UI) to reflect the current state.

        This method fills the display with a background color, draws a grid, snake(s), food, and scores.
        It also handles the display of the "Game Over" message when appropriate.

        Note: It is assumed that the method is called within a Pygame loop to update the display continuously.
        """
        self.display.fill((0, 0, 0))
        self.display.blit(self.background_image, (0, 0))

        # Draw vertical grid lines
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Red', (x, 0), (x, self.h), 1)

        # Draw horizontal grid lines
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Red', (0, y), (self.w, y), 1)

        # Draw snakes and food on the grid
        for snake_index in range(self.num_snakes):
            snake_color = self.snake_colors[snake_index]
            
            # Draw each segment of the snake
            for index, point in enumerate(self.snakes[snake_index]):
                if index == 0:  # Head of the snake (square)
                    pygame.draw.rect(
                        self.display,
                        snake_color,
                        pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE)
                    )
                else:  # Body of the snake (circle)
                    pygame.draw.circle(
                        self.display,
                        snake_color,
                        (point.x + BLOCK_SIZE // 2, point.y + BLOCK_SIZE // 2),
                        BLOCK_SIZE // 2
                    )

            # Draw food items on the grid
            for i, food in enumerate(self.food):
                food_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pygame.draw.rect(
                    self.display,
                    food_color,
                    pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE)
                )

            # Display "Game Over" message if the game is over
            if self.game_over[snake_index]:
                font = pygame.font.Font(None, 50)
                text = font.render("Game Over", True, (255, 255, 255))
                self.display.blit(text, (self.w // 2 - text.get_width() // 2, self.h // 2 - text.get_height() // 2))

            # Display scores on the UI
            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render("Score: " + str(self.score[snake_index]), True, (255, 255, 255))
            self.display.blit(score_text, (10, 10 + 40 * snake_index))

        pygame.display.flip()
        self.clock.tick(FPS)





    def handle_user_input(self):
        """
        Handles user input to control the snake's direction.

        Returns:
            list: A list representing the action taken by the human agent.

        This method checks for key presses and returns an action list based on the pressed keys.
        The action list represents the desired direction: [1, 0, 0] for no change, [0, 0, 1] for left turn, and [0, 1, 0] for right turn.
        """
        keys = pygame.key.get_pressed()
        
        # Check for the UP key
        if keys[pygame.K_UP]:
            return [0, 0, 1]  # Left turn action
        # Check for the DOWN key
        elif keys[pygame.K_DOWN]:
            return [0, 1, 0]  # Right turn action
        else:
            return [1, 0, 0]  # No change action



    def _move(self, actions):
        """
        Moves the snakes based on the specified actions.

        Parameters:
            actions (list): A list of actions representing the desired direction for each snake.

        This method updates the direction of each snake based on the provided actions.
        It then moves the snakes accordingly in the specified direction by updating their head positions.
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        # Iterate over each snake
        for snake_index in range(self.num_snakes):
            idx = clock_wise.index(self.directions[snake_index])

            # Determine the new direction based on the action
            if np.array_equal(actions[snake_index], [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(actions[snake_index], [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn (r -> d -> l -> u)
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn (r -> u -> l -> d)

            # Update the snake's direction
            self.directions[snake_index] = new_dir

            # Update the snake's head position based on the new direction
            x = self.heads[snake_index].x
            y = self.heads[snake_index].y
            if self.directions[snake_index] == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.directions[snake_index] == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.UP:
                y -= BLOCK_SIZE

            # Update the snake's head position
            self.heads[snake_index] = Point(x, y)



    def Create_agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma):
        """
        Creates an instance of the Agent class with specified parameters.

        Parameters:
            input_dim (int): Dimensionality of the input state.
            dim1 (int): Number of neurons in the first hidden layer.
            dim2 (int): Number of neurons in the second hidden layer.
            n_actions (int): Number of possible actions in the environment.
            lr (float): Learning rate for the agent.
            butch_size (int): Batch size for training the neural network.
            mem_size (int): Size of the replay memory for experience replay.
            gamma (float): Discount factor for future rewards.

        Returns:
            Agent: An instance of the Agent class with the specified parameters.
        """
        return Agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma)

    def plot(scores, mean_scores):
        """
        Plots the scores and mean scores during training.

        Parameters:
            scores (list): List of scores obtained in each game during training.
            mean_scores (list): List of mean scores over a window of games during training.

        This function creates a plot to visualize the training progress, showing scores and mean scores over the number of games.
        """
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, label='Score')
        plt.plot(mean_scores, label='Mean Score')
        plt.ylim(ymin=0)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        plt.legend()
        plt.show(block=False)
        plt.pause(.1)


if __name__ == '__main__':

    number_of_snakes   =  5  

    current_max = 0

    game = SnakeGame(num_snakes=number_of_snakes)

    agent = Agent(input_dimlist=[39, 39,39,39,39 ], fc1_dimlist=[512, 100,512, 100,512], fc2_dimlist=[256,256, 256,256,256],
                fc3_dimlist=[256, 256,256, 256,256], fc4_dimlist=[256, 256,256, 256,256], n_actions=3, losslist=[nn.MSELoss() , nn.MSELoss(),nn.MSELoss() , nn.MSELoss(),nn.MSELoss()], lrlist=[0.001, 0.001,0.001, 0.001,0.001],
                batch_size=[10,10,10,10,10], mem_size=[10000,10000,10000,10000,10000], gamma_list=[0.20, 0.10,0.20, 0.10,0.30], num_agents=5 , saved_path_list =["","","","",""] )
    
    # Run the game loop
    running = True
    step  =  [0]*game.num_snakes 

    TotalPlayerScorePro = [0]*game.num_snakes 
    Total_PlayedTime = [0]*game.num_snakes 
    TotalTimeBeforeDeath = [0]*game.num_snakes 
    TotalSnakeEaten   = [0]*game.num_snakes 
    TotalAppleEaten = [0]*game.num_snakes 

    TotalEatenSnakes = []
    for   _ in range(game.num_snakes) : 
        TotalEatenSnakes.append([])
    TotalEatenApples = []
    for   _ in range(game.num_snakes) : 
        TotalEatenApples.append([])
    Total_score_list = []
    for   _ in range(game.num_snakes) : 
        Total_score_list.append([])
    Total_Time_List  = []
    for   _ in range(game.num_snakes) : 
        Total_Time_List.append([])
    DataFrames = []
    BestPerformance = [0]*game.num_snakes
    for agent_idx in range(number_of_snakes):
        data = {
            f'n_games{agent_idx}': [],
            f'playerScoreProRound{agent_idx}': [],
            f'playerTotalScore{agent_idx}': [],
            f'TimePlayedPRoRound{agent_idx}': [],
            f'TotalTimePlayed{agent_idx}': [],
            f'MeanScore{agent_idx}': [],
            f'TimeOverScore{agent_idx}': [],
            f'TotalNumberofDeath{agent_idx}': [],
            f'TotalTimeSpendOverTotalTimeOfDeath{agent_idx}': [],
            f'Epsilon{agent_idx}': [],
            f'SnakeEatenProRound{agent_idx}': [],
            f'ApplesEatenProRound{agent_idx}': [],
            f'TotalSnakeEaten{agent_idx}': [],
            f'TotalApplesEaten{agent_idx}': [],
            f'TotalSnakeEatenOverTotalScore{agent_idx}': [],
            f'TotalApllesEatenOverTotalScore{agent_idx}': [],
            f'AvgAppleRate{agent_idx}': [], 
            f'AvgSnakesRate{agent_idx}' : [],
            f'CurrentState{agent_idx}': [],
        }
        
        DataFrames.append(data) 

    i = 0 

    while i <10000:

        
        old_states = game.get_state()
        actions = agent.choose_action(old_states)

        rewards, game_over, scores,  info  ,time_played ,apple_snake , DistanceToSnakesList ,DistanceToAppleList= game.play_step(actions)

        states_new = game.get_state()

        agent.short_mem(old_states, states_new, actions, rewards, game_over)

        screenshot_filename = f"/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/Coding/ENV1/GameImages/AllRound/AvoidCollison/screenshot{i}.png"
        if i >= 9900  : 
            pygame.image.save(game.display, screenshot_filename)


       
  
        if any(game_over) and not all(game_over):
            

            #print(f" info : {DistanceToAppleList}")
 
            indices = [index for index, value in enumerate(game_over) if value == True]     
            for index in indices:

                step[index]+=1  #number of round will be incremented 
                
                TotalPlayerScorePro[index]   = TotalPlayerScorePro[index] + scores[index]
                Total_PlayedTime[index] = time_played[index]+ Total_PlayedTime[index]
                TotalSnakeEaten[index]  = TotalSnakeEaten[index] +  apple_snake[index][1]
                #print(f"TotalSnakeEaten {index} : {TotalSnakeEaten[index]} ")


                TotalAppleEaten[index] =  TotalAppleEaten[index] +  apple_snake[index][0]
                DataFrames[index][f'n_games{index}'].append(step[index])  
                DataFrames[index][f'CurrentState{index}'].append(info[index])  
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index]/ step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'TimePlayedPRoRound{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalTimePlayed{index}'].append(Total_PlayedTime[index])
                if TotalPlayerScorePro[index]  > 0 :
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index]/TotalPlayerScorePro[index])#Time spend to reach a score 
                else  : 
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])   
                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TotalTimeSpendOverTotalTimeOfDeath{index}'].append(Total_PlayedTime[index]/step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])
                DataFrames[index][f'SnakeEatenProRound{index}'].append(TotalSnakeEaten[index])
                DataFrames[index][f'ApplesEatenProRound{index}'].append(TotalAppleEaten[index])
                DataFrames[index][f'TotalSnakeEaten{index}'].append(TotalSnakeEaten[index])
                DataFrames[index][f'TotalApplesEaten{index}'].append(TotalAppleEaten[index])
                #data frame  of the avg  appels  eaten over epochen 
                DataFrames[index][f'AvgAppleRate{index}'].append(TotalAppleEaten[index]/step[index]) #the  Apples Rate  or eaten 
                #avg snakes eaten    
                DataFrames[index][f'AvgSnakesRate{index}'].append(TotalSnakeEaten[index]/step[index]) # The snakes  Eatne or Rate  





                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'TotalSnakeEatenOverTotalScore{index}'].append(TotalSnakeEaten[index])
                else  : 
                    DataFrames[index][f'TotalSnakeEatenOverTotalScore{index}'].append(TotalSnakeEaten[index]/TotalPlayerScorePro[index])
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'TotalApllesEatenOverTotalScore{index}'].append(TotalAppleEaten[index])
                else  : 
                    DataFrames[index][f'TotalApllesEatenOverTotalScore{index}'].append(TotalAppleEaten[index]/TotalPlayerScorePro[index])
                   
            
                game.reset_snake(index)
                game._update_ui()
                game.clock.tick(SPEED)

            '''   
            for  agent_index  in range(game.num_snakes) :
               if   BestPerformance[agent_index] < scores[agent_index] : 
                   BestPerformance[agent_index] = scores[agent_index] 
                   agent.save(agent_index)
            '''
            
        



        # coding  Agnet  agains  human  

       

        

        elif all(game_over)  : 

            indices = [index for index, value in enumerate(game_over) if value == True]   
            for  index  in indices  :
                step[index]+=1 
                TotalPlayerScorePro[index]+= scores[index]
    
                Total_PlayedTime[index] += time_played[index]
     
                TotalSnakeEaten[index]  +=  apple_snake[index][1]
                #print(f" the eaten snake : {TotalSnakeEaten[index]}")

                TotalAppleEaten[index] +=  apple_snake[index][0]
                DataFrames[index][f'n_games{index}'].append(step[index])  

                DataFrames[index][f'CurrentState{index}'].append(info[index])  
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index]/ step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'TimePlayedPRoRound{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalTimePlayed{index}'].append(Total_PlayedTime[index])
                if TotalPlayerScorePro[index]  > 0 :
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index]/TotalPlayerScorePro[index])
                else  : 
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])   
                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TotalTimeSpendOverTotalTimeOfDeath{index}'].append(Total_PlayedTime[index]/step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])
                DataFrames[index][f'SnakeEatenProRound{index}'].append(apple_snake[index][0])
                DataFrames[index][f'ApplesEatenProRound{index}'].append(apple_snake[index][1])
                DataFrames[index][f'TotalSnakeEaten{index}'].append(TotalSnakeEaten[index])
                DataFrames[index][f'TotalApplesEaten{index}'].append(TotalAppleEaten[index])
                 #data frame  of the avg  appels  eaten over epochen 
                DataFrames[index][f'AvgAppleRate{index}'].append(TotalAppleEaten[index]/step[index]) #the  Apples Rate  or eaten 
                #avg snakes eaten    
                DataFrames[index][f'AvgSnakesRate{index}'].append(TotalSnakeEaten[index]/step[index]) # The snakes  Eatne or Rate  
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'TotalSnakeEatenOverTotalScore{index}'].append(TotalSnakeEaten[index])
                else  : 
                    DataFrames[index][f'TotalSnakeEatenOverTotalScore{index}'].append(TotalSnakeEaten[index]/TotalPlayerScorePro[index])
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'TotalApllesEatenOverTotalScore{index}'].append(TotalAppleEaten[index])
                else  : 
                    DataFrames[index][f'TotalApllesEatenOverTotalScore{index}'].append(TotalAppleEaten[index]/TotalPlayerScorePro[index])

          

            game.reset()

            game._update_ui() 

            game.clock.tick(SPEED)

            agent.long_mem() 
         
        #save the mdoell each 500 iterations  
        for  agent_index  in range(game.num_snakes) :
                if  i >= 9400 : 
                    if  i %  100 == 0   : 
                        agent.save(agent_index ,i )
        print(f"step : {i}")
        i+=1 

        game._update_ui()

        game.clock.tick(SPEED)
        #the total eaten snake is right  for both of  dthem 
        #print(f" the eaten snake : {TotalSnakeEaten}")


        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
       
    pygame.quit()


save_path = '/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/Coding/ENV1/ExcelFiles'


for dataFrame_index in range(game.num_snakes):
    df = pd.DataFrame(DataFrames[dataFrame_index])
    file_path = os.path.join(save_path, f'Test{dataFrame_index}.csv')  # Vollständiger Dateipfad
    df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))
