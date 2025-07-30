import random
from typing import List, Set, Tuple, Dict

import streamlit as st
from PIL import Image, ImageDraw
from math import atan2, cos, sin, pi

Cell = Tuple[int, int]



def _neighbors(grid: List[List[int]], cell: Cell) -> List[Cell]:
    """Return accessible neighbour cells from given cell."""
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
    """Find a path in the maze grid using BFS."""
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


def carve_corridor(grid: List[List[int]], start: Cell, end: Cell) -> List[Cell]:
    """Carve a corridor between two points avoiding existing passages."""
    h = (len(grid) - 1) // 2
    w = (len(grid[0]) - 1) // 2

    # grid for pathfinding where 1 means free to carve
    tmp = [[1 if grid[y][x] == 0 else 0 for x in range(len(grid[0]))] for y in range(len(grid))]
    sx, sy = start
    ex, ey = end
    tmp[sy * 2 + 1][sx * 2 + 1] = 1
    tmp[ey * 2 + 1][ex * 2 + 1] = 1
    path = _find_path(tmp, start, end)
    if not path:
        # fallback straight corridor
        x, y = sx, sy
        path = [(x, y)]
        while (x, y) != (ex, ey):
            if x < ex:
                x += 1
            elif x > ex:
                x -= 1
            elif y < ey:
                y += 1
            else:
                y -= 1
            path.append((x, y))

    # carve cells
    x0, y0 = path[0]
    grid[y0 * 2 + 1][x0 * 2 + 1] = 1
    for (x1, y1) in path[1:]:
        if x1 != x0:
            dx = 1 if x1 > x0 else -1
            grid[y0 * 2 + 1][x0 * 2 + 1 + dx] = 1
        if y1 != y0:
            dy = 1 if y1 > y0 else -1
            grid[y0 * 2 + 1 + dy][x0 * 2 + 1] = 1
        grid[y1 * 2 + 1][x1 * 2 + 1] = 1
        x0, y0 = x1, y1
    return path


def generate_maze(
    width: int, height: int, levels: int
) -> Tuple[
    List[List[List[int]]],
    List[Set[Cell]],
    List[Set[Cell]],
    List[Tuple[int, int, int]],
    Cell,
    Cell,
]:
    """Generate a 3D maze where the main path can move up or down."""

    grids = [[[0] * (width * 2 + 1) for _ in range(height * 2 + 1)] for _ in range(levels)]
    up = [set() for _ in range(levels)]
    down = [set() for _ in range(levels)]

    path: List[Tuple[int, int, int]] = []

    level = levels - 1
    sx, sy = random.randrange(width), random.randrange(height)
    start = (level, sx, sy)
    path.append(start)
    grids[level][sy * 2 + 1][sx * 2 + 1] = 1

    x, y = sx, sy
    while level > 0:
        move_choices = []
        if level > 0:
            move_choices.append(-1)
        if level < levels - 1:
            move_choices.append(1)
        move = random.choice(move_choices)
        new_level = max(0, min(levels - 1, level + move))

        nx, ny = random.randrange(width), random.randrange(height)
        segment = carve_corridor(grids[level], (x, y), (nx, ny))
        for cx, cy in segment[1:]:
            path.append((level, cx, cy))

        if new_level < level:
            down[level].add((nx, ny))
            up[new_level].add((nx, ny))
        else:
            up[level].add((nx, ny))
            down[new_level].add((nx, ny))

        level = new_level
        x, y = nx, ny
        path.append((level, x, y))
        grids[level][y * 2 + 1][x * 2 + 1] = 1

    fx, fy = width // 2, height - 1
    segment = carve_corridor(grids[0], (x, y), (fx, fy))
    for cx, cy in segment[1:]:
        path.append((0, cx, cy))

    finish = (0, fx, fy)
    path.append(finish)

    return grids, up, down, path, start, finish


def draw_level(
    grids: List[List[List[int]]],
    up: List[Set[Cell]],
    down: List[Set[Cell]],
    level: int,
    cell_size: int = 30,
    wall: int = 2,
    solution: List[Cell] | None = None,
    start: Tuple[int, int, int] | None = None,
    finish: Tuple[int, int, int] | None = None,
) -> Image.Image:
    grid = grids[level]
    maze_h = (len(grid) - 1) // 2
    maze_w = (len(grid[0]) - 1) // 2
    img = Image.new("RGB", (maze_w * cell_size + wall, maze_h * cell_size + wall), "white")
    draw = ImageDraw.Draw(img)

    # draw outer borders
    draw.rectangle([0, 0, maze_w * cell_size, maze_h * cell_size], outline="black", width=wall)

    # vertical walls
    for y in range(maze_h):
        for x in range(maze_w):
            cy = 2 * y + 1
            cx = 2 * x + 1
            if grid[cy][cx + 1] == 0:  # wall to the right
                x_pix = (x + 1) * cell_size
                y1 = y * cell_size
                y2 = (y + 1) * cell_size
                draw.line([(x_pix, y1), (x_pix, y2)], fill="black", width=wall)
            if grid[cy + 1][cx] == 0:  # wall below
                y_pix = (y + 1) * cell_size
                x1 = x * cell_size
                x2 = (x + 1) * cell_size
                draw.line([(x1, y_pix), (x2, y_pix)], fill="black", width=wall)
    if solution and len(solution) > 1:
        pts_px = [
            (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
            for x, y in solution
        ]
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
    radius = cell_size // 3
    for ex, ey in up[level]:
        cx = ex * 2 + 1
        cy = ey * 2 + 1
        draw.ellipse(
            [
                (cx * cell_size // 2 - radius, cy * cell_size // 2 - radius),
                (cx * cell_size // 2 + radius, cy * cell_size // 2 + radius),
            ],
            fill="white",
            outline="black",
        )
    for ex, ey in down[level]:
        cx = ex * 2 + 1
        cy = ey * 2 + 1
        draw.ellipse(
            [
                (cx * cell_size // 2 - radius, cy * cell_size // 2 - radius),
                (cx * cell_size // 2 + radius, cy * cell_size // 2 + radius),
            ],
            fill="black",
        )
    if start and start[0] == level:
        sx, sy = start[1], start[2]
        draw.text(
            (sx * cell_size + cell_size // 3, sy * cell_size + cell_size // 3),
            "S",
            fill="blue",
        )
    if finish and finish[0] == level:
        fx, fy = finish[1], finish[2]
        draw.text(
            (fx * cell_size + cell_size // 3, fy * cell_size + cell_size // 3),
            "F",
            fill="green",
        )
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
