import random
from typing import List, Set, Tuple

import streamlit as st
from PIL import Image, ImageDraw

Cell = Tuple[int, int]


def _corridor(path: List[Cell], start: Cell, end: Cell) -> List[Cell]:
    """Return a simple manhattan corridor from start to end."""
    x1, y1 = start
    x2, y2 = end
    cells = []
    step = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step, step):
        cells.append((x, y1))
    step = 1 if y2 >= y1 else -1
    for y in range(y1, y2 + step, step):
        cells.append((x2, y))
    if cells and path and cells[0] == path[-1]:
        cells = cells[1:]
    path.extend(cells)
    return path


def generate_maze(width: int, height: int, levels: int, max_elev: int):
    grid = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]
    up = [set() for _ in range(levels)]
    down = [set() for _ in range(levels)]
    path: List[Tuple[int, int, int]] = []

    x = random.randrange(width)
    y = random.randrange(height)
    level = 0
    grid[level][y][x] = 1
    path.append((level, x, y))

    while level < levels - 1:
        nx, ny = random.randrange(width), random.randrange(height)
        corridor: List[Cell] = []
        _corridor(corridor, (x, y), (nx, ny))
        for cx, cy in corridor:
            grid[level][cy][cx] = 1
            path.append((level, cx, cy))
        up[level].add((nx, ny))
        down[level + 1].add((nx, ny))
        grid[level + 1][ny][nx] = 1
        path.append((level + 1, nx, ny))
        x, y = nx, ny
        level += 1

    gx, gy = random.randrange(width), random.randrange(height)
    corridor: List[Cell] = []
    _corridor(corridor, (x, y), (gx, gy))
    for cx, cy in corridor:
        grid[level][cy][cx] = 1
        path.append((level, cx, cy))

    for lv in range(levels - 1):
        current = len(up[lv])
        while current < max_elev:
            ex, ey = random.randrange(width), random.randrange(height)
            if (ex, ey) in up[lv]:
                continue
            up[lv].add((ex, ey))
            down[lv + 1].add((ex, ey))
            grid[lv][ey][ex] = 1
            grid[lv + 1][ey][ex] = 1
            current += 1

    return grid, up, down, path, (gx, gy)


def draw_level(grid, up, down, level: int, cell_size: int = 25, solution: Set[Cell] | None = None) -> Image.Image:
    h = len(grid[level])
    w = len(grid[level][0])
    img = Image.new("RGB", (w * cell_size, h * cell_size), "black")
    draw = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            if grid[level][y][x]:
                color = "white"
                if solution and (x, y) in solution:
                    color = "red"
                draw.rectangle(
                    [x * cell_size, y * cell_size, (x + 1) * cell_size - 1, (y + 1) * cell_size - 1],
                    fill=color,
                )
    radius = cell_size // 4
    for ex, ey in up[level]:
        cx = ex * cell_size + cell_size // 2
        cy = ey * cell_size + cell_size // 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill="white",
            outline="black",
        )
    for ex, ey in down[level]:
        cx = ex * cell_size + cell_size // 2
        cy = ey * cell_size + cell_size // 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill="black",
        )
    return img


def main() -> None:
    st.title("3D Maze")
    with st.sidebar:
        w = st.number_input("width", 5, 30, 10)
        h = st.number_input("height", 5, 30, 10)
        z = st.number_input("levels", 2, 10, 3)
        max_e = st.number_input("max elevators", 2, 10, 2, step=2)
        generate = st.button("Generate")

    if generate or "maze" not in st.session_state:
        grid, up, down, path, goal = generate_maze(int(w), int(h), int(z), int(max_e))
        st.session_state["maze"] = (grid, up, down, path, goal)
        st.session_state["level"] = 0
        st.session_state["show"] = False

    grid, up, down, path, goal = st.session_state["maze"]
    level = st.session_state.get("level", 0)
    show = st.session_state.get("show", False)

    c1, c2, c3 = st.columns(3)
    if c1.button("Up") and level < len(grid) - 1:
        st.session_state["level"] = level + 1
        level += 1
    if c2.button("Down") and level > 0:
        st.session_state["level"] = level - 1
        level -= 1
    if c3.button("Toggle solution"):
        st.session_state["show"] = not show
        show = st.session_state["show"]

    st.subheader(f"Floor {level}")
    sol_set: Set[Cell] | None = None
    if show:
        sol_set = {(x, y) for lv, x, y in path if lv == level}
    img = draw_level(grid, up, down, level, solution=sol_set)
    st.image(img)


if __name__ == "__main__":
    main()
