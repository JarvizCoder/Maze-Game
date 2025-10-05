# main.py
import pygame
import random
import heapq
from typing import List, Tuple, Set, Optional, Dict
from enum import Enum

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    
    @staticmethod
    def opposite(direction: 'Direction') -> 'Direction':
        opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST
        }
        return opposites[direction]

class Cell:
    """Represents a single cell in the maze with walls and position information."""
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.walls = {direction: True for direction in Direction}
        self.visited = False
        self.is_start = False
        self.is_end = False
    
    def remove_wall(self, direction: Direction) -> None:
        """Remove the wall in the specified direction."""
        self.walls[direction] = False
    
    def has_wall(self, direction: Direction) -> bool:
        """Check if there's a wall in the specified direction."""
        return self.walls.get(direction, True)

class Maze:
    """Maze generator and solver using various algorithms."""
    
    def __init__(self, width: int, height: int, cell_size: int = 20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cells = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self.start = (0, 0)
        self.end = (width - 1, height - 1)
        self._generate_maze()
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (within bounds)."""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors
    
    def _get_connected_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get neighbors that are connected (no wall in between)."""
        connected = []
        for direction in Direction:
            dx, dy = direction.value
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                not self.cells[x][y].has_wall(direction)):
                connected.append((nx, ny))
        return connected
    
    def _generate_maze(self) -> None:
        """Generate maze using recursive backtracking algorithm."""
        stack = [(0, 0)]
        self.cells[0][0].visited = True
        
        while stack:
            x, y = stack[-1]
            current = self.cells[x][y]
            neighbors = [(nx, ny) for nx, ny in self._get_neighbors(x, y) 
                        if not self.cells[nx][ny].visited]
            
            if not neighbors:
                stack.pop()
                continue
                
            nx, ny = random.choice(neighbors)
            neighbor = self.cells[nx][ny]
            
            # Remove walls between current and neighbor
            direction = Direction((nx - x, ny - y))
            current.remove_wall(direction)
            neighbor.remove_wall(Direction.opposite(direction))
            
            neighbor.visited = True
            stack.append((nx, ny))
        
        # Mark start and end positions
        self.cells[self.start[0]][self.start[1]].is_start = True
        self.cells[self.end[0]][self.end[1]].is_end = True
    
    def solve_bfs(self) -> List[Tuple[int, int]]:
        """Solve the maze using Breadth-First Search."""
        queue = [(self.start, [self.start])]
        visited = set([self.start])
        
        while queue:
            (x, y), path = queue.pop(0)
            
            if (x, y) == self.end:
                return path
                
            for nx, ny in self._get_connected_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return []
    
    def solve_dfs(self) -> List[Tuple[int, int]]:
        """Solve the maze using Depth-First Search."""
        stack = [(self.start, [self.start])]
        visited = set([self.start])
        
        while stack:
            (x, y), path = stack.pop()
            
            if (x, y) == self.end:
                return path
                
            for nx, ny in self._get_connected_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append(((nx, ny), path + [(nx, ny)]))
        return []
    
    def solve_astar(self) -> List[Tuple[int, int]]:
        """Solve the maze using A* algorithm."""
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: heuristic(self.start, self.end)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == self.end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]
                
            for neighbor in self._get_connected_neighbors(*current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, self.end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []

class MazeGame:
    """Main game class handling user interaction and visualization."""
    
    def __init__(self, width: int = 20, height: int = 15, cell_size: int = 30):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size + 100  # Extra space for UI
        self.screen_height = height * cell_size + 100
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Solver")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.reset_game()
    
    def reset_game(self) -> None:
        """Reset the game state with a new maze."""
        self.maze = Maze(self.width, self.height, self.cell_size)
        self.player_pos = list(self.maze.start)
        self.path = []
        self.solution = []
        self.lives = 3
        self.game_over = False
        self.show_solution = False
        self.solving_algorithm = "BFS"  # Default algorithm
    
    def draw(self) -> None:
        """Draw the maze, player, and UI elements."""
        self.screen.fill((255, 255, 255))
        
        # Draw maze
        for x in range(self.width):
            for y in range(self.height):
                cell = self.maze.cells[x][y]
                rect = pygame.Rect(
                    x * self.cell_size + 50, 
                    y * self.cell_size + 50, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # Draw cell background
                if cell.is_start:
                    pygame.draw.rect(self.screen, (144, 238, 144), rect)  # Light green
                elif cell.is_end:
                    pygame.draw.rect(self.screen, (255, 182, 193), rect)  # Light red
                elif (x, y) == tuple(self.player_pos):
                    pygame.draw.rect(self.screen, (100, 100, 255), rect)  # Blue for player
                elif (x, y) in self.path:
                    pygame.draw.rect(self.screen, (200, 200, 255), rect)  # Light blue for path
                elif self.show_solution and (x, y) in self.solution:
                    pygame.draw.rect(self.screen, (255, 255, 200), rect)  # Light yellow for solution
                
                # Draw walls
                if cell.has_wall(Direction.NORTH):
                    pygame.draw.line(
                        self.screen, (0, 0, 0), 
                        (x * self.cell_size + 50, y * self.cell_size + 50),
                        ((x + 1) * self.cell_size + 50, y * self.cell_size + 50), 2
                    )
                if cell.has_wall(Direction.SOUTH):
                    pygame.draw.line(
                        self.screen, (0, 0, 0), 
                        (x * self.cell_size + 50, (y + 1) * self.cell_size + 50),
                        ((x + 1) * self.cell_size + 50, (y + 1) * self.cell_size + 50), 2
                    )
                if cell.has_wall(Direction.WEST):
                    pygame.draw.line(
                        self.screen, (0, 0, 0), 
                        (x * self.cell_size + 50, y * self.cell_size + 50),
                        (x * self.cell_size + 50, (y + 1) * self.cell_size + 50), 2
                    )
                if cell.has_wall(Direction.EAST):
                    pygame.draw.line(
                        self.screen, (0, 0, 0), 
                        ((x + 1) * self.cell_size + 50, y * self.cell_size + 50),
                        ((x + 1) * self.cell_size + 50, (y + 1) * self.cell_size + 50), 2
                    )
        
        # Draw UI
        lives_text = self.font.render(f"Lives: {self.lives}", True, (0, 0, 0))
        self.screen.blit(lives_text, (20, 20))
        
        if self.game_over:
            if self.lives > 0:
                game_over_text = self.font.render("You Win! Press R to restart", True, (0, 200, 0))
            else:
                game_over_text = self.font.render("Game Over! Press R to restart", True, (200, 0, 0))
            self.screen.blit(game_over_text, (self.screen_width // 2 - 100, 20))
        
        # Draw algorithm selection
        algo_text = self.font.render(f"Algorithm: {self.solving_algorithm}", True, (0, 0, 0))
        self.screen.blit(algo_text, (self.screen_width - 200, 20))
        
        pygame.display.flip()
    
    def move_player(self, dx: int, dy: int) -> None:
        """Move the player if the move is valid."""
        if self.game_over:
            return
            
        x, y = self.player_pos
        new_x, new_y = x + dx, y + dy
        
        # Check if move is within bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            return
        
        # Check for walls
        if dx > 0 and self.maze.cells[x][y].has_wall(Direction.EAST):
            return
        if dx < 0 and self.maze.cells[x][y].has_wall(Direction.WEST):
            return
        if dy > 0 and self.maze.cells[x][y].has_wall(Direction.SOUTH):
            return
        if dy < 0 and self.maze.cells[x][y].has_wall(Direction.NORTH):
            return
        
        self.player_pos = [new_x, new_y]
        self.path.append((x, y))
        
        # Check if reached the end
        if (new_x, new_y) == self.maze.end:
            self.game_over = True
    
    def solve_maze(self) -> None:
        """Solve the maze using the selected algorithm."""
        if self.solving_algorithm == "BFS":
            self.solution = self.maze.solve_bfs()
        elif self.solving_algorithm == "DFS":
            self.solution = self.maze.solve_dfs()
        elif self.solving_algorithm == "A*":
            self.solution = self.maze.solve_astar()
        self.show_solution = True
    
    def run(self) -> None:
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif not self.game_over:
                        if event.key == pygame.K_UP:
                            self.move_player(0, -1)
                        elif event.key == pygame.K_DOWN:
                            self.move_player(0, 1)
                        elif event.key == pygame.K_LEFT:
                            self.move_player(-1, 0)
                        elif event.key == pygame.K_RIGHT:
                            self.move_player(1, 0)
                        elif event.key == pygame.K_s:
                            self.solve_maze()
                        elif event.key == pygame.K_1:
                            self.solving_algorithm = "BFS"
                            self.show_solution = False
                            self.solution = []
                        elif event.key == pygame.K_2:
                            self.solving_algorithm = "DFS"
                            self.show_solution = False
                            self.solution = []
                        elif event.key == pygame.K_3:
                            self.solving_algorithm = "A*"
                            self.show_solution = False
                            self.solution = []
            
            # Check if player hit a wall
            if not self.game_over and len(self.path) > 1:
                x, y = self.player_pos
                if (x, y) in self.path[:-1]:  # Player backtracked
                    if (x, y) in self.path[:-1][-5:]:  # Only penalize if backtracking too much
                        self.lives -= 1
                        if self.lives <= 0:
                            self.game_over = True
                        self.path = self.path[:self.path.index((x, y)) + 1]  # Truncate path
                elif len(self.path) > self.width * self.height * 2:  # Prevent infinite loops
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    self.path = []
                    self.player_pos = list(self.maze.start)
            
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    # Create and run the game with a 20x15 maze
    game = MazeGame(40, 35, 20)
    game.run()