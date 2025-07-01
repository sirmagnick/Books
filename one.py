"""Simple pipe maze generator with a Streamlit interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

import streamlit as st
from PIL import Image, ImageDraw


Direction = str


@dataclass
class Cell:
    """Represents a single maze cell with open pipe directions."""

    opens: Set[Direction]


def generate_maze(width: int, height: int) -> Dict[Tuple[int, int], Cell]:
    """Generate a perfect maze using a depth-first search algorithm."""

    directions: Dict[Direction, Tuple[int, int]] = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0),
    }
    opposite: Dict[Direction, Direction] = {"N": "S", "E": "W", "S": "N", "W": "E"}

    grid: Dict[Tuple[int, int], Cell] = {
        (x, y): Cell(set()) for y in range(height) for x in range(width)
    }

    stack: list[Tuple[int, int]] = [(0, 0)]
    visited = {(0, 0)}
    while stack:
        x, y = stack[-1]
        neighbors = []
        for d, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                neighbors.append((d, nx, ny))

        if neighbors:
            d, nx, ny = random.choice(neighbors)
            grid[(x, y)].opens.add(d)
            grid[(nx, ny)].opens.add(opposite[d])
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    return grid


def draw_maze(
    grid: Dict[Tuple[int, int], Cell],
    width: int,
    height: int,
    cell_size: int = 60,
    pipe_width: int = 20,
) -> Image.Image:
    """Render the maze as a Pillow image."""

    img = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(img)

    for (x, y), cell in grid.items():
        cx = x * cell_size + cell_size // 2
        cy = y * cell_size + cell_size // 2
        half = cell_size // 2
        if "N" in cell.opens:
            draw.line((cx, cy, cx, cy - half), fill="blue", width=pipe_width)
        if "S" in cell.opens:
            draw.line((cx, cy, cx, cy + half), fill="blue", width=pipe_width)
        if "E" in cell.opens:
            draw.line((cx, cy, cx + half, cy), fill="blue", width=pipe_width)
        if "W" in cell.opens:
            draw.line((cx, cy, cx - half, cy), fill="blue", width=pipe_width)

    return img


def main() -> None:
    st.title("Pipe Maze Generator")
    st.sidebar.write("Maze Settings")
    width = st.sidebar.slider("Width", 5, 20, 10)
    height = st.sidebar.slider("Height", 5, 20, 10)

    if st.button("Generate Maze"):
        grid = generate_maze(width, height)
        st.session_state["pipe_maze_grid"] = (width, height, grid)

    if "pipe_maze_grid" in st.session_state:
        w, h, grid = st.session_state["pipe_maze_grid"]
        img = draw_maze(grid, w, h)
        st.image(img)
    else:
        st.write("Click 'Generate Maze' to create a new maze.")


if __name__ == "__main__":
    main()

