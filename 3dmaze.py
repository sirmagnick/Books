import random
from typing import List, Set, Tuple

import streamlit as st
from PIL import Image, ImageDraw
from math import atan2, cos, sin, pi

Cell = Tuple[int, int]


def _corridor(start: Cell, end: Cell) -> List[Cell]:
    """Return a simple manhattan corridor from start to end."""
    x1, y1 = start
    x2, y2 = end
    path: List[Cell] = []
    step = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step, step):
        path.append((x, y1))
    step = 1 if y2 >= y1 else -1
    for y in range(y1, y2 + step, step):
        path.append((x2, y))
    return path


def _generate_level(width: int, height: int) -> List[List[int]]:
    """Generate a simple maze for one level using recursive backtracker."""
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


def generate_maze(width: int, height: int, levels: int) -> Tuple[List[List[List[int]]], List[Set[Cell]], List[Set[Cell]], List[Tuple[int, int, int]], Cell, Cell]:
    grids = [_generate_level(width, height) for _ in range(levels)]
    up = [set() for _ in range(levels)]
    down = [set() for _ in range(levels)]
    path: List[Tuple[int, int, int]] = []

    sx, sy = random.randrange(width), random.randrange(height)
    start = (levels - 1, sx, sy)
    path.append(start)

    x, y = sx, sy
    for lv in range(levels - 1, 0, -1):
        nx, ny = random.randrange(width), random.randrange(height)
        for cx, cy in _corridor((x, y), (nx, ny)):
            grids[lv][cy * 2 + 1][cx * 2 + 1] = 1
            path.append((lv, cx, cy))
        down[lv].add((nx, ny))
        up[lv - 1].add((nx, ny))
        x, y = nx, ny
        path.append((lv - 1, x, y))

    fx, fy = width // 2, height - 1
    for cx, cy in _corridor((x, y), (fx, fy)):
        grids[0][cy * 2 + 1][cx * 2 + 1] = 1
        path.append((0, cx, cy))

    finish = (0, fx, fy)
    path.append(finish)
    return grids, up, down, path, start, finish


def draw_level(grids: List[List[List[int]]], up: List[Set[Cell]], down: List[Set[Cell]], level: int, cell_size: int = 20, solution: List[Cell] | None = None, start: Tuple[int,int,int]|None=None, finish: Tuple[int,int,int]|None=None) -> Image.Image:
    grid = grids[level]
    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGB", (w * cell_size, h * cell_size), "black")
    draw = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            if grid[y][x]:
                draw.rectangle([x * cell_size, y * cell_size, (x+1)*cell_size-1, (y+1)*cell_size-1], fill="white")
    if solution and len(solution) > 1:
        pts = [(x * 2 + 1, y * 2 + 1) for x, y in solution]
        # convert to pixel centers
        pts_px = [(px * cell_size //2 + cell_size//2, py * cell_size //2 + cell_size//2) for px,py in pts]
        draw.line(pts_px, fill="red", width=3)
        x1,y1 = pts_px[-2]
        x2,y2 = pts_px[-1]
        angle = atan2(y2-y1, x2-x1)
        size = cell_size//2
        arrow = [
            (x2, y2),
            (x2 - size*cos(angle - pi/6), y2 - size*sin(angle - pi/6)),
            (x2 - size*cos(angle + pi/6), y2 - size*sin(angle + pi/6)),
        ]
        draw.polygon(arrow, fill="red")
    radius = cell_size // 3
    for ex, ey in up[level]:
        cx = ex * 2 + 1
        cy = ey * 2 + 1
        draw.ellipse([(cx*cell_size//2 - radius, cy*cell_size//2 - radius), (cx*cell_size//2 + radius, cy*cell_size//2 + radius)], fill="white", outline="black")
    for ex, ey in down[level]:
        cx = ex * 2 + 1
        cy = ey * 2 + 1
        draw.ellipse([(cx*cell_size//2 - radius, cy*cell_size//2 - radius), (cx*cell_size//2 + radius, cy*cell_size//2 + radius)], fill="black")
    if start and start[0]==level:
        sx, sy = start[1], start[2]
        draw.text(((sx*2+1)*cell_size//2, (sy*2+1)*cell_size//2), "S", fill="blue")
    if finish and finish[0]==level:
        fx, fy = finish[1], finish[2]
        draw.text(((fx*2+1)*cell_size//2, (fy*2+1)*cell_size//2), "F", fill="green")
    return img


def main() -> None:
    st.title("3D Maze")
    with st.sidebar:
        w = st.number_input("width", 5, 40, 20)
        h = st.number_input("height", 5, 40, 20)
        z = st.number_input("levels", 10, 100, 20)
        generate = st.button("Generate")

    if generate or "maze" not in st.session_state:
        grids, up, down, path, start, finish = generate_maze(int(w), int(h), int(z))
        st.session_state["maze"] = (grids, up, down, path, start, finish)
        st.session_state["level"] = z-1
        st.session_state["show"] = False

    grids, up, down, path, start, finish = st.session_state["maze"]
    level = st.session_state.get("level", len(grids)-1)
    show = st.session_state.get("show", False)

    c1, c2, c3 = st.columns(3)
    if c1.button("Up") and level < len(grids) - 1:
        level += 1
        st.session_state["level"] = level
    if c2.button("Down") and level > 0:
        level -= 1
        st.session_state["level"] = level
    if c3.button("Toggle solution"):
        show = not show
        st.session_state["show"] = show

    st.subheader(f"Floor {level}")
    sol_cells: List[Cell] | None = None
    if show:
        sol_cells = [ (x,y) for lv,x,y in path if lv==level ]
    img = draw_level(grids, up, down, level, solution=sol_cells, start=start, finish=finish)
    st.image(img)


if __name__ == "__main__":
    main()
