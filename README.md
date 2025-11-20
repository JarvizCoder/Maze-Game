# Maze Solver & Visualizer

This project is a comprehensive maze generation and solving tool. It features a fully functional desktop application built with Pygame that allows users to generate, navigate, and visualize the solving of mazes. Additionally, it includes the foundational structure for a pygame based UI design.

## Installation & Running

There are two main components to this project: the Pygame desktop application and the React web application.

#### 1. Desktop Maze Game (Pygame)

This is a complete, playable maze game.

1.  **Navigate to the application directory:**
    ```bash
    cd Mazeapp/mazeapp-react
    ```

2.  **Install Pygame:**
    ```bash
    pip install pygame
    ```

3.  **Run the game:**
    ```bash
    python main.py
    ```

**How to Play:**

*   **Arrow Keys:** Move the player through the maze.
*   **S Key:** Start/Stop the automatic solver visualization.
*   **1, 2, 3 Keys:** Switch between solving algorithms (1: BFS, 2: DFS, 3: A*).
*   **R Key:** Reset the game and generate a new maze.

## Features

*   **Random Maze Generation:** Creates a new, random maze every time the application is run.
*   **Interactive Gameplay:** Navigate the maze from a start point to an end point.
*   **Pathfinding Algorithm Visualization:** Watch different algorithms solve the maze in real-time.
    *   Breadth-First Search (BFS)
    *   Depth-First Search (DFS)
    *   A* Search
*   **React Frontend:** A basic structure for a web-based visualization tool is included.

## Technologies Used

*   **Desktop Application:**
    *   Python
    *   Pygame
*   **Web Application:**
    *   React
    *   Vite

## Project Structure

```
Maze-Project/
├── Mazeapp/
│   ├── mazeapp-react/
│   │   ├── src/         # React frontend source code
│   │   ├── main.py      # Pygame application
│   │   └── ...
│   └── ...
└── README.md
```


## Future Development

The project is set up to evolve into a full-stack web application where the Python backend (maze generation and solving logic) communicates with the React frontend. The current empty `server/stream_server.py` file indicates the planned location for the API that will stream maze data to the web interface.
