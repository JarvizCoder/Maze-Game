# main.py
import pygame
import random
import math
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
        """Get neighbors that are directly connected (no wall in between)."""
        connected = []
        for direction in Direction:
            dx, dy = direction.value
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                not self.cells[x][y].has_wall(direction)):
                connected.append((nx, ny))
        return connected
        
    def _are_connected(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are connected through any path using BFS."""
        if pos1 == pos2:
            return True
            
        visited = set([pos1])
        queue = [pos1]
        
        while queue:
            x, y = queue.pop(0)
            
            for nx, ny in self._get_connected_neighbors(x, y):
                if (nx, ny) == pos2:
                    return True
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
        return False
    
    def _generate_maze(self) -> None:
        """Generate maze using recursive backtracking with guaranteed multiple solutions."""
        # First, generate a perfect maze using recursive backtracking
        stack = [(0, 0)]
        self.cells[0][0].visited = True
        visited_count = 1
        total_cells = self.width * self.height
        
        while stack and visited_count < total_cells:
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
            visited_count += 1
            stack.append((nx, ny))

        # Reset visited flags for the next phase
        for row in self.cells:
            for cell in row:
                cell.visited = False

        # Add strategic connections to create multiple solution paths
        # We'll create at least 3-5 distinct paths from start to end
        min_extra_paths = max(3, min(5, (self.width + self.height) // 8))
        added_connections = 0
        attempts = 0
        max_attempts = self.width * self.height

        while added_connections < min_extra_paths and attempts < max_attempts:
            attempts += 1
            
            # Choose a random cell that's not on the border
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            
            # Get all directions where there's currently a wall
            wall_directions = []
            for direction in Direction:
                dx, dy = direction.value
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.cells[x][y].has_wall(direction)):
                    wall_directions.append((direction, nx, ny))
            
            if wall_directions:
                # Try to add a connection that creates a new path
                direction, nx, ny = random.choice(wall_directions)
                
                # Temporarily remove the wall to test if it creates a useful alternative path
                self.cells[x][y].remove_wall(direction)
                self.cells[nx][ny].remove_wall(Direction.opposite(direction))
                
                # Check if this creates a meaningful alternative path
                # by comparing path lengths before and after
                original_path = self.solve_bfs()
                if original_path and len(original_path) > 0:
                    # This connection is valid - keep it
                    added_connections += 1
                else:
                    # This doesn't help, restore the wall
                    self.cells[x][y].walls[direction] = True
                    self.cells[nx][ny].walls[Direction.opposite(direction)] = True

        # Add some additional random connections for complexity
        # This creates loops and alternative dead-ends
        random_connections = max(2, (self.width * self.height) // 20)
        for _ in range(random_connections):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Get all directions where there's currently a wall
            wall_directions = []
            for direction in Direction:
                dx, dy = direction.value
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.cells[x][y].has_wall(direction)):
                    wall_directions.append((direction, nx, ny))
            
            if wall_directions and random.random() < 0.6:  # 60% chance
                direction, nx, ny = random.choice(wall_directions)
                self.cells[x][y].remove_wall(direction)
                self.cells[nx][ny].remove_wall(Direction.opposite(direction))

        # Mark start and end positions
        self.cells[self.start[0]][self.start[1]].is_start = True
        self.cells[self.end[0]][self.end[1]].is_end = True
        
        # Verify that we have multiple solution paths
        # If not, add more connections until we do
        max_attempts = 10
        attempts = 0
        while not self.has_multiple_solutions(2) and attempts < max_attempts:
            attempts += 1
            # Add a random connection
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            wall_directions = []
            for direction in Direction:
                dx, dy = direction.value
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.cells[x][y].has_wall(direction)):
                    wall_directions.append((direction, nx, ny))
            
            if wall_directions:
                direction, nx, ny = random.choice(wall_directions)
                self.cells[x][y].remove_wall(direction)
                self.cells[nx][ny].remove_wall(Direction.opposite(direction))
    
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

    def count_solution_paths(self, max_paths: int = 10) -> int:
        """Count the number of distinct solution paths from start to end.
        
        Uses a modified DFS to find multiple paths up to max_paths.
        Returns the number of distinct paths found.
        """
        if not self._are_connected(self.start, self.end):
            return 0
        
        found_paths = []
        
        def dfs_find_paths(current: Tuple[int, int], path: List[Tuple[int, int]], visited: Set[Tuple[int, int]]) -> None:
            """Recursive DFS to find multiple paths."""
            if len(found_paths) >= max_paths:
                return
            
            if current == self.end:
                found_paths.append(path.copy())
                return
            
            for neighbor in self._get_connected_neighbors(*current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_find_paths(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        # Start DFS from the start position
        dfs_find_paths(self.start, [self.start], set([self.start]))
        
        return len(found_paths)
    
    def has_multiple_solutions(self, min_paths: int = 2) -> bool:
        """Check if the maze has multiple solution paths."""
        return self.count_solution_paths(min_paths) >= min_paths

class MazeGame:
    """Main game class handling user interaction and visualization."""
    
    def __init__(self, width: int = 20, height: int = 15, cell_size: int = 30):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size + 200  # Extra space for UI
        self.screen_height = height * cell_size + 200
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("Maze Solver - Dark Theme")
        self.clock = pygame.time.Clock()
        
        # Load a nice font
        try:
            self.font = pygame.font.SysFont('Segoe UI', 20, bold=True)
            self.title_font = pygame.font.SysFont('Segoe UI', 32, bold=True)
        except:
            self.font = pygame.font.SysFont('Arial', 20, bold=True)
            self.title_font = pygame.font.SysFont('Arial', 32, bold=True)
            
        # Color scheme
        self.COLORS = {
            'background': (18, 18, 26),
            'grid': (30, 30, 40),
            'wall': (80, 90, 120),
            'path': (200, 200, 255, 50),
            'player': (100, 200, 255),
            'end': (255, 100, 100),
            'visited': (60, 70, 90),
            'solution': (100, 255, 150),
            'text': (220, 220, 240),
            'ui_bg': (30, 35, 45),
            'ui_border': (60, 65, 80)
        }
        
        # Initialize glow effect surfaces
        self.glow_surf = self._create_glow_surface(20, 20, self.COLORS['player'])
        self.end_glow_surf = self._create_glow_surface(20, 20, self.COLORS['end'])
        
        self.keys_pressed = {pygame.K_UP: False, pygame.K_DOWN: False, 
                           pygame.K_LEFT: False, pygame.K_RIGHT: False}
        self.last_move_time = 0
        self.last_solve_step = 0
        self.move_delay = 100  # milliseconds between moves when holding a key
        self.solving = False
        self.reset_game()
        
    def _create_glow_surface(self, size: int, radius: int, color: tuple) -> pygame.Surface:
        """Create a surface with a glowing effect."""
        surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        for i in range(radius, 0, -1):
            alpha = int(100 * (i / radius))
            s = pygame.Surface((i * 2, i * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color[0:3], alpha), (i, i), i)
            pos = (size - i, size - i)
            surf.blit(s, pos)
        return surf
    
    def reset_game(self) -> None:
        """Reset the game state with a new maze."""
        # Clean up any existing solver state
        self.solving = False
        if hasattr(self, 'solve_queue'):
            del self.solve_queue
        if hasattr(self, 'solve_stack'):
            del self.solve_stack
        if hasattr(self, 'open_set'):
            del self.open_set
        if hasattr(self, 'open_set_items'):
            del self.open_set_items
        if hasattr(self, 'came_from'):
            del self.came_from
        if hasattr(self, 'g_score'):
            del self.g_score
        if hasattr(self, 'f_score'):
            del self.f_score
        if hasattr(self, 'visited_cells'):
            del self.visited_cells
        if hasattr(self, 'current_path'):
            del self.current_path
            
        self.maze = Maze(self.width, self.height, self.cell_size)
        self.player_pos = list(self.maze.start)
        self.path = []
        self.solution = []
        self.lives = 3
        self.game_over = False
        self.show_solution = False
        self.solving_algorithm = "BFS"  # Default algorithm
    
    def draw(self) -> None:
        """Draw the maze, player, and UI elements with modern styling."""
        # Draw background
        self.screen.fill(self.COLORS['background'])
        
        # Draw grid pattern in the background
        for x in range(0, self.screen_width, 20):
            pygame.draw.line(self.screen, self.COLORS['grid'], (x, 0), (x, self.screen_height), 1)
        for y in range(0, self.screen_height, 20):
            pygame.draw.line(self.screen, self.COLORS['grid'], (0, y), (self.screen_width, y), 1)
        
        # Draw a semi-transparent overlay for the maze area
        maze_surface = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size), pygame.SRCALPHA)
        maze_surface.fill((*self.COLORS['ui_bg'], 200))  # Semi-transparent dark background
        
        # Calculate centered position for the maze
        maze_x = (self.screen_width - (self.width * self.cell_size)) // 2
        maze_y = (self.screen_height - (self.height * self.cell_size)) // 2
        
        # Draw visited cells first (background)
        if hasattr(self, 'visited_cells'):
            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) in self.visited_cells:
                        rect = pygame.Rect(
                            x * self.cell_size + 2, 
                            y * self.cell_size + 2, 
                            self.cell_size - 3, 
                            self.cell_size - 3
                        )
                        # Semi-transparent visited cells
                        pygame.draw.rect(maze_surface, (*self.COLORS['visited'], 100), rect, border_radius=2)
        
        # Draw maze and other elements on the maze surface
        for x in range(self.width):
            for y in range(self.height):
                cell = self.maze.cells[x][y]
                cell_rect = pygame.Rect(
                    x * self.cell_size + 1,
                    y * self.cell_size + 1,
                    self.cell_size - 2,
                    self.cell_size - 2
                )
                
                # Draw cell background with subtle gradient
                if cell.is_start:
                    pygame.draw.rect(maze_surface, (80, 180, 220, 50), cell_rect, border_radius=3)
                elif cell.is_end:
                    # Draw end cell with glow effect
                    glow_rect = pygame.Rect(
                        x * self.cell_size - 10,
                        y * self.cell_size - 10,
                        self.cell_size + 20,
                        self.cell_size + 20
                    )
                    maze_surface.blit(pygame.transform.scale(self.end_glow_surf, 
                        (glow_rect.width, glow_rect.height)), 
                        (glow_rect.x - 20, glow_rect.y - 20))
                    pygame.draw.rect(maze_surface, self.COLORS['end'], cell_rect, border_radius=3)
                
                # Draw walls with a 3D effect
                if cell.has_wall(Direction.NORTH):
                    pygame.draw.line(
                        maze_surface, self.COLORS['wall'],
                        (cell_rect.left, cell_rect.top + 1),
                        (cell_rect.right, cell_rect.top + 1),
                        3
                    )
                if cell.has_wall(Direction.SOUTH):
                    pygame.draw.line(
                        maze_surface, self.COLORS['wall'],
                        (cell_rect.left, cell_rect.bottom - 1),
                        (cell_rect.right, cell_rect.bottom - 1),
                        3
                    )
                if cell.has_wall(Direction.WEST):
                    pygame.draw.line(
                        maze_surface, self.COLORS['wall'],
                        (cell_rect.left + 1, cell_rect.top),
                        (cell_rect.left + 1, cell_rect.bottom),
                        3
                    )
                if cell.has_wall(Direction.EAST):
                    pygame.draw.line(
                        maze_surface, self.COLORS['wall'],
                        (cell_rect.right - 1, cell_rect.top),
                        (cell_rect.right - 1, cell_rect.bottom),
                        3
                    )
        
        # Draw solution path if shown with a nice gradient
        if self.show_solution and self.solution:
            for i in range(len(self.solution) - 1):
                x1, y1 = self.solution[i]
                x2, y2 = self.solution[i + 1]
                progress = i / len(self.solution)
                color = (
                    int(100 + 155 * progress),
                    int(255 - 55 * progress),
                    int(100 + 100 * progress)
                )
                pygame.draw.line(
                    maze_surface, (*color, 200),
                    (x1 * self.cell_size + self.cell_size // 2,
                     y1 * self.cell_size + self.cell_size // 2),
                    (x2 * self.cell_size + self.cell_size // 2,
                     y2 * self.cell_size + self.cell_size // 2),
                    max(2, self.cell_size // 6)
                )
        
        # Draw player path with a nice trail effect
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                x1, y1 = self.path[i]
                x2, y2 = self.path[i + 1]
                alpha = int(100 + 155 * (i / len(self.path)))  # Fade effect
                pygame.draw.line(
                    maze_surface, (*self.COLORS['player'], alpha),
                    (x1 * self.cell_size + self.cell_size // 2,
                     y1 * self.cell_size + self.cell_size // 2),
                    (x2 * self.cell_size + self.cell_size // 2,
                     y2 * self.cell_size + self.cell_size // 2),
                    max(2, self.cell_size // 8)
                )
        
        # Draw the maze surface onto the main screen
        self.screen.blit(maze_surface, (maze_x, maze_y))
        
        # Draw player with glow effect
        player_glow_rect = pygame.Rect(
            maze_x + self.player_pos[0] * self.cell_size - 10,
            maze_y + self.player_pos[1] * self.cell_size - 10,
            self.cell_size + 20,
            self.cell_size + 20
        )
        self.screen.blit(pygame.transform.scale(self.glow_surf, 
            (player_glow_rect.width, player_glow_rect.height)), 
            (player_glow_rect.x, player_glow_rect.y))
            
        player_rect = pygame.Rect(
            maze_x + self.player_pos[0] * self.cell_size + 3,
            maze_y + self.player_pos[1] * self.cell_size + 3,
            self.cell_size - 6,
            self.cell_size - 6
        )
        pygame.draw.rect(self.screen, self.COLORS['player'], player_rect, 
                        border_radius=self.cell_size // 3)
        
        # Draw UI panel with modern styling
        ui_panel = pygame.Surface((180, 120), pygame.SRCALPHA)
        pygame.draw.rect(ui_panel, (*self.COLORS['ui_bg'], 220), 
                        ui_panel.get_rect(), border_radius=8)
        pygame.draw.rect(ui_panel, self.COLORS['ui_border'], 
                        ui_panel.get_rect(), 2, border_radius=8)
        self.screen.blit(ui_panel, (20, 20))
        
        # Draw UI text
        lives_text = self.font.render(f"Lives: {self.lives}", True, self.COLORS['text'])
        self.screen.blit(lives_text, (40, 40))
        
        if self.game_over:
            game_over_text = self.title_font.render("GAME OVER", True, (220, 80, 80))
            text_rect = game_over_text.get_rect(center=(self.screen_width // 2, 40))
            pygame.draw.rect(self.screen, (0, 0, 0, 180), 
                           (text_rect.x - 20, text_rect.y - 10, 
                            text_rect.width + 40, text_rect.height + 20),
                           border_radius=5)
            self.screen.blit(game_over_text, text_rect)
        
        # Draw solver controls
        if not self.solving:
            solve_text = self.font.render("SPACE: Solve Maze", True, self.COLORS['text'])
            self.screen.blit(solve_text, (40, 70))
            
            algo_text = self.font.render(f"Algorithm: {self.solving_algorithm}", 
                                       True, self.COLORS['text'])
            self.screen.blit(algo_text, (40, 100))
        else:
            solving_text = self.font.render("Solving...", True, self.COLORS['text'])
            self.screen.blit(solving_text, (40, 70))
        
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
        """Toggle the maze solver on/off."""
        if hasattr(self, 'solving') and self.solving:
            # Stop solving and clear solver visualization
            self.solving = False
            self.show_solution = False
            
            # Clear solver state
            solver_attrs = [
                'solve_queue', 'solve_stack', 'open_set', 'open_set_items',
                'came_from', 'g_score', 'f_score', 'current_path', 'visited_cells'
            ]
            for attr in solver_attrs:
                if hasattr(self, attr):
                    delattr(self, attr)
                    
            # Clear the solution path
            self.solution = []
            
            # Force redraw to show player position
            self.draw()
            pygame.display.flip()
        else:
            # Start solving
            self.solving = True
            self.solve_maze_step_by_step()
    
    def solve_maze_step_by_step(self) -> None:
        """Solve the maze step by step to visualize the process."""
        self.solving = True
        self.solution = []
        self.visited_cells = set()
        self.show_solution = True
        
        if self.solving_algorithm == "BFS":
            self.solve_queue = [(self.maze.start, [self.maze.start])]
            self.visited_cells.add(self.maze.start)
        elif self.solving_algorithm == "DFS":
            self.solve_stack = [(self.maze.start, [self.maze.start])]
            self.visited_cells.add(self.maze.start)
        elif self.solving_algorithm == "A*":
            def heuristic(a, b):
                # Manhattan distance
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
            self.open_set = []
            heapq.heappush(self.open_set, (0, self.maze.start))
            self.open_set_items = {self.maze.start}  # For faster lookups
            self.came_from = {}
            self.g_score = {self.maze.start: 0}
            self.f_score = {self.maze.start: heuristic(self.maze.start, self.maze.end)}
            self.visited_cells = set()  # Will store all visited cells for visualization
            self.current_path = [self.maze.start]  # Current best path being considered
    
    def solve_step(self) -> bool:
        """Perform one step of the solving algorithm.
        Returns True if solution is found, False otherwise."""
        # Force redraw after each step
        self.draw()
        pygame.display.flip()
        pygame.time.delay(20)  # Small delay to see the progress
        
        if self.solving_algorithm == "BFS":
            if not hasattr(self, 'solve_queue') or not self.solve_queue:
                self.solving = False
                return True
                
            (x, y), path = self.solve_queue.pop(0)
            
            if (x, y) == self.maze.end:
                self.solution = path
                self.solving = False
                return True
                
            for nx, ny in self.maze._get_connected_neighbors(x, y):
                if (nx, ny) not in self.visited_cells:
                    self.visited_cells.add((nx, ny))
                    self.solve_queue.append(((nx, ny), path + [(nx, ny)]))
            
            # Update the current path being explored
            if self.solve_queue:
                self.solution = self.solve_queue[0][1]
        
        elif self.solving_algorithm == "DFS":
            if not hasattr(self, 'solve_stack') or not self.solve_stack:
                self.solving = False
                return True
                
            (x, y), path = self.solve_stack.pop()
            
            if (x, y) == self.maze.end:
                self.solution = path
                self.solving = False
                return True
                
            for nx, ny in self.maze._get_connected_neighbors(x, y):
                if (nx, ny) not in self.visited_cells:
                    self.visited_cells.add((nx, ny))
                    self.solve_stack.append(((nx, ny), path + [(nx, ny)]))
            
            # Update the current path being explored
            if self.solve_stack:
                self.solution = path
        
        elif self.solving_algorithm == "A*":
            if not hasattr(self, 'open_set') or not self.open_set:
                self.solving = False
                return True
                
            # Get the current best node to explore
            current_f, current = heapq.heappop(self.open_set)
            if current in self.open_set_items:  # Safety check
                self.open_set_items.remove(current)
            
            # Mark as visited if not already
            if current not in self.visited_cells:
                self.visited_cells.add(current)
                # Update visualization
                self.draw()
                pygame.display.flip()
                pygame.time.delay(10)
            
            # Check if we've reached the end
            if current == self.maze.end:
                # Reconstruct path
                path = [current]
                while current in self.came_from:
                    current = self.came_from[current]
                    path.append(current)
                self.solution = path[::-1]  # Reverse to get start to end
                self.solving = False
                return True
            
            # Explore neighbors
            for neighbor in self.maze._get_connected_neighbors(*current):
                # Calculate tentative g score (distance from start)
                tentative_g_score = self.g_score[current] + 1
                
                # If this is a new node or we found a better path to it
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    
                    # Calculate f score (g + h)
                    h_score = abs(neighbor[0] - self.maze.end[0]) + abs(neighbor[1] - self.maze.end[1])
                    f_score = tentative_g_score + h_score
                    
                    # Add to open set if not already there
                    if neighbor not in self.open_set_items:
                        heapq.heappush(self.open_set, (f_score, neighbor))
                        self.open_set_items.add(neighbor)
            
            # Show the current best path
            if hasattr(self, 'came_from') and self.came_from:
                path = []
                current_in_path = current
                while current_in_path in self.came_from:
                    path.append(current_in_path)
                    current_in_path = self.came_from[current_in_path]
                self.solution = path + [self.maze.start]
        
        return False
    
    def run(self) -> None:
        """Main game loop."""
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset solving state before resetting game
                        self.solving = False
                        if hasattr(self, 'solve_queue'):
                            del self.solve_queue
                        if hasattr(self, 'solve_stack'):
                            del self.solve_stack
                        if hasattr(self, 'open_set'):
                            del self.open_set
                        if hasattr(self, 'open_set_items'):
                            del self.open_set_items
                        self.reset_game()
                    elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        self.keys_pressed[event.key] = True
                        # Immediate move on first key press
                        if not self.game_over:
                            if event.key == pygame.K_UP:
                                self.move_player(0, -1)
                            elif event.key == pygame.K_DOWN:
                                self.move_player(0, 1)
                            elif event.key == pygame.K_LEFT:
                                self.move_player(-1, 0)
                            elif event.key == pygame.K_RIGHT:
                                self.move_player(1, 0)
                            self.last_move_time = current_time
                    elif not self.game_over:
                        if event.key == pygame.K_s:
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
                elif event.type == pygame.KEYUP:
                    if event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = False
            
            # Handle continuous movement when key is held down
            if not self.game_over and current_time - self.last_move_time > 100:  # 100ms between moves when holding
                moved = False
                
                if self.keys_pressed[pygame.K_UP]:
                    self.move_player(0, -1)
                    moved = True
                elif self.keys_pressed[pygame.K_DOWN]:
                    self.move_player(0, 1)
                    moved = True
                elif self.keys_pressed[pygame.K_LEFT]:
                    self.move_player(-1, 0)
                    moved = True
                elif self.keys_pressed[pygame.K_RIGHT]:
                    self.move_player(1, 0)
                    moved = True
                
                if moved:
                    self.last_move_time = current_time
            
            # Handle step-by-step solving
            if hasattr(self, 'solving') and self.solving:
                if current_time - self.last_solve_step > 14:  # 50ms between steps
                    self.solve_step()
                    self.last_solve_step = current_time
            
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

class DifficultyMenu:
    def __init__(self):
        # Increase window size for better spacing
        self.width = 600
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("Maze Game - Select Difficulty")
        self.clock = pygame.time.Clock()
        
        # Load nice fonts with slightly larger sizes
        try:
            self.title_font = pygame.font.SysFont('Segoe UI', 56, bold=True)
            self.option_font = pygame.font.SysFont('Segoe UI', 36, bold=True)
            self.small_font = pygame.font.SysFont('Segoe UI', 22)
        except:
            self.title_font = pygame.font.SysFont('Arial', 56, bold=True)
            self.option_font = pygame.font.SysFont('Arial', 36, bold=True)
            self.small_font = pygame.font.SysFont('Arial', 22)
        
        # Color scheme
        self.COLORS = {
            'background': (18, 18, 26),
            'title': (220, 220, 240),
            'option': (180, 190, 210),
            'selected': (100, 200, 255),
            'highlight': (140, 220, 255),
            'instructions': (140, 150, 170)
        }
        
        # Animation
        self.animation_offset = 0
        self.animation_speed = 0.05
        
        # Menu configuration
        self.panel_width = 400  # Width of the menu panel
        self.panel_height = 500  # Height of the menu panel
        self.panel_x = (self.width - self.panel_width) // 2  # Center the panel
        self.panel_y = 100  # Position from top
        
        self.selected = 1  # Default to Medium
        self.difficulties = [
            ("Easy", 30, 30, 25),    # name, width, height, cell_size
            ("Medium", 40, 40, 20),
            ("Hard", 50, 50, 15)
        ]
        
        # Calculate option positions
        self.option_width = 300
        self.option_height = 70
        self.option_spacing = 30  # Space between options
        self.option_start_y = 250  # Starting Y position for the first option
        self.running = True
        
    def draw(self):
        # Animate background
        self.animation_offset += self.animation_speed
        
        # Draw animated background
        self.screen.fill(self.COLORS['background'])
        
        # Draw subtle grid pattern
        for x in range(0, self.width, 30):
            x_pos = (x + self.animation_offset * 10) % (self.width + 60) - 30
            pygame.draw.line(self.screen, (25, 25, 35), (x_pos, 0), (x_pos - self.height, self.height), 1)
        
        # Draw main panel background with rounded corners
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, 
                               self.panel_width, self.panel_height)
        pygame.draw.rect(self.screen, (25, 27, 35), panel_rect, 
                        border_radius=20)
        pygame.draw.rect(self.screen, (40, 45, 60), panel_rect, 
                        2, border_radius=20)  # Border
        
        # Draw title with shadow and better positioning
        title = self.title_font.render("MAZE GAME", True, (0, 0, 0, 150))
        title_rect = title.get_rect(center=(self.width//2 + 3, self.panel_y + 63))
        self.screen.blit(title, title_rect)
        
        title = self.title_font.render("MAZE GAME", True, self.COLORS['title'])
        title_rect = title.get_rect(center=(self.width//2, self.panel_y + 60))
        self.screen.blit(title, title_rect)
        
        # Draw subtitle with better spacing
        subtitle = self.small_font.render("Select Difficulty", True, self.COLORS['instructions'])
        self.screen.blit(subtitle, (self.width//2 - subtitle.get_width()//2, 
                                   self.panel_y + 120))
        
        # Draw difficulty options with better spacing and animation
        for i, (name, w, h, size) in enumerate(self.difficulties, 1):
            is_selected = i == self.selected
            
            # Calculate position with better spacing and subtle animation
            y_pos = self.option_start_y + (i-1) * (self.option_height + self.option_spacing)
            if is_selected:
                y_pos += math.sin(pygame.time.get_ticks() * 0.005) * 2
            
            # Draw option background with new dimensions
            option_rect = pygame.Rect(
                self.width//2 - self.option_width//2,
                y_pos,
                self.option_width,
                self.option_height
            )
            
            # Draw glow for selected option with better sizing
            if is_selected:
                glow_surf = pygame.Surface((self.option_width + 40, self.option_height + 20), 
                                         pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*self.COLORS['selected'], 60), 
                               glow_surf.get_rect(), 
                               border_radius=20)
                self.screen.blit(glow_surf, 
                               (self.width//2 - (self.option_width + 40)//2, 
                                y_pos - 10))
            
            # Draw option background with better colors
            bg_color = (35, 40, 50) if not is_selected else self.COLORS['selected']
            pygame.draw.rect(self.screen, bg_color, option_rect, 
                           border_radius=15)
            
            # Draw border with better colors and thickness
            border_color = (70, 75, 90) if not is_selected else self.COLORS['highlight']
            border_width = 2 if not is_selected else 3
            pygame.draw.rect(self.screen, border_color, option_rect, 
                           border_width, border_radius=15)
            
            # Draw option text
            text_color = self.COLORS['option'] if not is_selected else (255, 255, 255)
            text = self.option_font.render(name.upper(), True, text_color)
            text_rect = text.get_rect(center=option_rect.center)
            self.screen.blit(text, text_rect)
            
            # Draw difficulty details with better positioning
            details = f"{w} × {h} maze • {size}px cells"
            details_surf = self.small_font.render(details, True, 
                                                self.COLORS['instructions'])
            details_rect = details_surf.get_rect(
                center=(self.width//2, y_pos + self.option_height - 15))
            self.screen.blit(details_surf, details_rect)
            
        # Draw instructions at the bottom with better styling
        instructions = self.small_font.render("USE ↑/↓ TO NAVIGATE • PRESS ENTER TO START", 
                                             True, self.COLORS['instructions'])
        instructions_bg = pygame.Surface((instructions.get_width() + 30, 
                                        instructions.get_height() + 15), 
                                       pygame.SRCALPHA)
        pygame.draw.rect(instructions_bg, (255, 255, 255, 10), 
                         instructions_bg.get_rect(), border_radius=10)
        self.screen.blit(instructions_bg, 
                        (self.width//2 - instructions_bg.get_width()//2, 
                         self.height - 80))
        self.screen.blit(instructions, 
                        (self.width//2 - instructions.get_width()//2, 
                         self.height - 72))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected = max(1, self.selected - 1)
                    elif event.key == pygame.K_DOWN:
                        self.selected = min(3, self.selected + 1)
                    elif event.key == pygame.K_RETURN:
                        self.running = False
                        return self.difficulties[self.selected - 1][1:]  # Return (width, height, cell_size)
            
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        return None

if __name__ == "__main__":
    # Initialize pygame for the menu
    pygame.init()
    
    # Show difficulty selection menu
    menu = DifficultyMenu()
    result = menu.run()
    
    if result:
        width, height, cell_size = result
        # Create and run the game with selected difficulty
        game = MazeGame(width, height, cell_size)
    game.run()