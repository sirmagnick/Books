import random
from typing import List, Tuple, Dict, Set

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


def _generate_level(width: int, height: int, allowed: Set[Cell]) -> List[List[int]]:
    w, h = width, height
    grid = [[0] * (w * 2 + 1) for _ in range(h * 2 + 1)]
    for x, y in allowed:
        grid[y * 2 + 1][x * 2 + 1] = 1
    if not allowed:
        return grid
    start_cell = random.choice(list(allowed))
    stack = [start_cell]
    visited = {start_cell}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        x, y = stack[-1]
        neigh = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < w
                and 0 <= ny < h
                and (nx, ny) in allowed
                and (nx, ny) not in visited
            ):
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


def _straight_path(start: Cell, end: Cell) -> List[Cell]:
    """Return a list of cells forming a simple Manhattan path."""
    x0, y0 = start
    x1, y1 = end
    path: List[Cell] = []
    if random.choice([True, False]):
        step = 1 if x1 >= x0 else -1
        for x in range(x0, x1 + step, step):
            path.append((x, y0))
        step = 1 if y1 >= y0 else -1
        for y in range(y0 + step, y1 + step, step):
            path.append((x1, y))
    else:
        step = 1 if y1 >= y0 else -1
        for y in range(y0, y1 + step, step):
            path.append((x0, y))
        step = 1 if x1 >= x0 else -1
        for x in range(x0 + step, x1 + step, step):
            path.append((x, y1))
    return path


def generate_multi_maze(width: int, height: int, paths: int):
    paths = max(1, paths)
    starts: List[Cell] = []
    finishes: List[Cell] = []
    lines: List[List[Cell]] = []
    used: Set[Cell] = set()
    attempts = 0
    while len(lines) < paths and attempts < 500:
        start = (random.randrange(width), random.randrange(height))
        finish = (random.randrange(width), random.randrange(height))
        if start == finish:
            attempts += 1
            continue
        line = _straight_path(start, finish)
        if any(c in used for c in line):
            attempts += 1
            continue
        lines.append(line)
        starts.append(start)
        finishes.append(finish)
        used.update(line)
    paths = len(lines)

    # assign cells to regions using multi-source BFS
    region = [[-1 for _ in range(height)] for _ in range(width)]
    from collections import deque
    q = deque()
    for idx, line in enumerate(lines):
        for x, y in line:
            region[x][y] = idx
            q.append((x, y, idx))
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        x, y, idx = q.popleft()
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and region[nx][ny] == -1:
                region[nx][ny] = idx
                q.append((nx, ny, idx))

    regions: List[Set[Cell]] = [set() for _ in range(paths)]
    for x in range(width):
        for y in range(height):
            idx = region[x][y]
            if idx != -1 and (x, y) not in lines[idx]:
                regions[idx].add((x, y))

    full_grid = [[0] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    for cells in regions:
        if not cells:
            continue
        reg_grid = _generate_level(width, height, cells)
        for y in range(height * 2 + 1):
            for x in range(width * 2 + 1):
                if reg_grid[y][x]:
                    full_grid[y][x] = 1

    # carve corridors
    for line in lines:
        prev = None
        for x, y in line:
            full_grid[y * 2 + 1][x * 2 + 1] = 1
            if prev:
                dx = x - prev[0]
                dy = y - prev[1]
                full_grid[prev[1] * 2 + 1 + dy][prev[0] * 2 + 1 + dx] = 1
            prev = (x, y)

    solutions = [line for line in lines]
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
