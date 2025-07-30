import random
from typing import List, Tuple, Dict

import streamlit as st
from PIL import Image, ImageDraw
from math import atan2, cos, sin, pi

Cell = Tuple[int, int]


def _neighbors(grid: List[List[int]], cell: Cell) -> List[Cell]:
    x, y = cell
    w = (len(grid[0]) - 1) // 2
    h = (len(grid) - 1) // 2
    neigh = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            if grid[y * 2 + 1 + dy][x * 2 + 1 + dx] == 1:
                neigh.append((nx, ny))
    return neigh


def _find_path(grid: List[List[int]], start: Cell, end: Cell) -> List[Cell]:
    queue = [start]
    prev: Dict[Cell, Cell | None] = {start: None}
    while queue:
        cur = queue.pop(0)
        if cur == end:
            break
        for n in _neighbors(grid, cur):
            if n not in prev:
                prev[n] = cur
                queue.append(n)
    if end not in prev:
        return []
    path: List[Cell] = []
    cur: Cell | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _generate_level(width: int, height: int) -> List[List[int]]:
    w, h = width, height
    grid = [[0] * (w * 2 + 1) for _ in range(h * 2 + 1)]
    for y in range(h):
        for x in range(w):
            grid[y * 2 + 1][x * 2 + 1] = 1
    stack = [(random.randrange(w), random.randrange(h))]
    visited = {stack[0]}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        x, y = stack[-1]
        neigh = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                neigh.append((dx, dy, nx, ny))
        if neigh:
            dx, dy, nx, ny = random.choice(neigh)
            grid[y * 2 + 1 + dy][x * 2 + 1 + dx] = 1
            grid[ny * 2 + 1][nx * 2 + 1] = 1
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    return grid


def generate_multi_maze(width: int, height: int, paths: int):
    paths = max(1, paths)
    sub_w = width // paths
    grids = []
    starts = []
    finishes = []
    solutions = []
    offset_x = 0
    full_grid = [[0] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    for i in range(paths):
        w = sub_w if i < paths - 1 else width - sub_w * (paths - 1)
        g = _generate_level(w, height)
        grids.append(g)
        start = (random.randrange(w), random.randrange(height))
        finish = (random.randrange(w), random.randrange(height))
        while finish == start:
            finish = (random.randrange(w), random.randrange(height))
        path = _find_path(g, start, finish)
        starts.append((start[0] + offset_x, start[1]))
        finishes.append((finish[0] + offset_x, finish[1]))
        solutions.append([(x + offset_x, y) for x, y in path])
        # copy g into full_grid
        for y in range(len(g)):
            for x in range(len(g[0])):
                full_grid[y][x + offset_x * 2] = g[y][x]
        # add dividing wall between regions except last
        if i < paths - 1:
            for y in range(height * 2 + 1):
                full_grid[y][offset_x * 2 + w * 2] = 0
        offset_x += w
    return full_grid, starts, finishes, solutions


def draw_maze(grid: List[List[int]], starts, finishes, solutions, cell_size=30, wall=2):
    maze_h = (len(grid) - 1) // 2
    maze_w = (len(grid[0]) - 1) // 2
    img = Image.new("RGB", (maze_w * cell_size + wall, maze_h * cell_size + wall), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, maze_w * cell_size, maze_h * cell_size], outline="black", width=wall)
    for y in range(maze_h):
        for x in range(maze_w):
            cy = 2 * y + 1
            cx = 2 * x + 1
            if grid[cy][cx + 1] == 0:
                x_pix = (x + 1) * cell_size
                draw.line([(x_pix, y * cell_size), (x_pix, (y + 1) * cell_size)], fill="black", width=wall)
            if grid[cy + 1][cx] == 0:
                y_pix = (y + 1) * cell_size
                draw.line([(x * cell_size, y_pix), ((x + 1) * cell_size, y_pix)], fill="black", width=wall)
    for path in solutions:
        if len(path) > 1:
            pts_px = [(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2) for x, y in path]
            draw.line(pts_px, fill="red", width=3)
            x1, y1 = pts_px[-2]
            x2, y2 = pts_px[-1]
            angle = atan2(y2 - y1, x2 - x1)
            size = cell_size // 2
            arrow = [
                (x2, y2),
                (x2 - size*cos(angle - pi/6), y2 - size*sin(angle - pi/6)),
                (x2 - size*cos(angle + pi/6), y2 - size*sin(angle + pi/6)),
            ]
            draw.polygon(arrow, fill="red")
    for sx, sy in starts:
        draw.text((sx * cell_size + cell_size // 3, sy * cell_size + cell_size // 3), "S", fill="blue")
    for fx, fy in finishes:
        draw.text((fx * cell_size + cell_size // 3, fy * cell_size + cell_size // 3), "F", fill="green")
    return img


def main() -> None:
    st.title("Multi Maze")
    with st.sidebar:
        w = st.number_input("width", 5, 60, 20)
        h = st.number_input("height", 5, 40, 20)
        p = st.number_input("ilość ścieżek", 1, 10, 2)
        generate = st.button("Generate")
    if generate or "multi_maze" not in st.session_state:
        grid, starts, finishes, solutions = generate_multi_maze(int(w), int(h), int(p))
        st.session_state["multi_maze"] = (grid, starts, finishes, solutions)
    grid, starts, finishes, solutions = st.session_state["multi_maze"]
    img = draw_maze(grid, starts, finishes, solutions)
    st.image(img)


if __name__ == "__main__":
    main()
