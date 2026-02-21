"""Pipe Maze Generator - 2D and 2.5D styles.

Generates pipe-style mazes where corridors are drawn as thick black lines.
In 2.5D mode, crossings show one pipe passing under another.
"""

import random
import io

import streamlit as st
from PIL import Image, ImageDraw

# ── Direction bit flags ───────────────────────────────────────────────────────
N, S, E, W = 1, 2, 4, 8
OPP = {N: S, S: N, E: W, W: E}
DXDY = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}


# ── Maze generation ───────────────────────────────────────────────────────────

def generate_maze(width: int, height: int) -> list[list[int]]:
    """Generate a maze using iterative DFS.

    Returns a 2D grid where grid[y][x] is a bitmask of open passages.
    """
    grid = [[0] * width for _ in range(height)]
    visited = [[False] * width for _ in range(height)]
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        x, y = stack[-1]
        dirs = [
            d for d in (N, S, E, W)
            if 0 <= x + DXDY[d][0] < width
            and 0 <= y + DXDY[d][1] < height
            and not visited[y + DXDY[d][1]][x + DXDY[d][0]]
        ]
        if dirs:
            d = random.choice(dirs)
            dx, dy = DXDY[d]
            grid[y][x] |= d
            grid[y + dy][x + dx] |= OPP[d]
            visited[y + dy][x + dx] = True
            stack.append((x + dx, y + dy))
        else:
            stack.pop()

    return grid


def add_crossings(
    grid: list[list[int]],
    width: int,
    height: int,
    n: int,
) -> set[tuple[int, int]]:
    """Add extra passages to create over/under crossings for the 2.5D effect.

    Finds interior cells that already have a clean straight-through passage in
    one axis and adds the perpendicular passage, creating an H ∩ V crossing.
    Neighbouring crossings are avoided to prevent visual clutter.

    Returns the set of crossing cell coordinates.
    """
    crossings: set[tuple[int, int]] = set()

    # Collect candidate cells: straight-through in exactly one axis
    candidates: list[tuple[int, int]] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            cell = grid[y][x]
            straight_h = bool(cell & E) and bool(cell & W) and not (cell & (N | S))
            straight_v = bool(cell & N) and bool(cell & S) and not (cell & (E | W))
            if straight_h or straight_v:
                candidates.append((x, y))

    random.shuffle(candidates)

    for x, y in candidates:
        if len(crossings) >= n:
            break
        if (x, y) in crossings:
            continue
        # Skip if adjacent to another crossing
        if any(
            (x + dx, y + dy) in crossings
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        ):
            continue

        cell = grid[y][x]
        if bool(cell & E) and bool(cell & W) and not (cell & (N | S)):
            # Existing H passage → add N-S passage
            grid[y][x] |= N | S
            grid[y - 1][x] |= S
            grid[y + 1][x] |= N
            crossings.add((x, y))
        elif bool(cell & N) and bool(cell & S) and not (cell & (E | W)):
            # Existing V passage → add E-W passage
            grid[y][x] |= E | W
            grid[y][x - 1] |= E
            grid[y][x + 1] |= W
            crossings.add((x, y))

    return crossings


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _center(x: int, y: int, cell_size: int) -> tuple[int, int]:
    half = cell_size // 2
    return x * cell_size + half, y * cell_size + half


def _seg(draw: ImageDraw.ImageDraw, cx, cy, d, half, line_w, color):
    """Draw a pipe segment from (cx, cy) toward direction d."""
    dx, dy = DXDY[d]
    draw.line([(cx, cy), (cx + dx * half, cy + dy * half)], fill=color, width=line_w)


# ── 2D rendering ──────────────────────────────────────────────────────────────

def draw_2d(
    grid: list[list[int]],
    cell_size: int = 40,
    line_w: int = 4,
) -> Image.Image:
    """Render a 2D pipe maze – all pipes drawn at the same visual level."""
    height, width = len(grid), len(grid[0])
    black = (0, 0, 0, 255)
    img = Image.new("RGBA", (width * cell_size, height * cell_size), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    half = cell_size // 2
    r = max(1, line_w // 2)

    for y in range(height):
        for x in range(width):
            cx, cy = _center(x, y, cell_size)
            cell = grid[y][x]
            for direction in (N, S, E, W):
                if cell & direction:
                    _seg(d, cx, cy, direction, half, line_w, black)
            d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=black)

    return img


# ── 2.5D rendering ────────────────────────────────────────────────────────────

def draw_25d(
    grid: list[list[int]],
    crossings: set[tuple[int, int]],
    cell_size: int = 40,
    line_w: int = 4,
) -> Image.Image:
    """Render a 2.5D pipe maze.

    At crossing cells, horizontal pipes are drawn on top of vertical pipes.
    A white gap is cut through the vertical pipe at the crossing point,
    creating the illusion that the horizontal pipe passes over it.

    Rendering order:
    1. All vertical segments (go under at crossings)
    2. Joint dots for non-crossing cells
    3. White eraser gap over vertical pipe at each crossing
    4. All horizontal segments (go over at crossings)
    5. Joint dots for crossing cells (on top of H pipe)
    """
    height, width = len(grid), len(grid[0])
    black = (0, 0, 0, 255)
    white = (255, 255, 255, 255)
    img = Image.new("RGBA", (width * cell_size, height * cell_size), white)
    d = ImageDraw.Draw(img)
    half = cell_size // 2
    r = max(1, line_w // 2)
    gap = line_w + 2  # half-height of the erased zone

    # 1. Vertical segments
    for y in range(height):
        for x in range(width):
            cx, cy = _center(x, y, cell_size)
            cell = grid[y][x]
            if cell & N:
                d.line([(cx, cy), (cx, cy - half)], fill=black, width=line_w)
            if cell & S:
                d.line([(cx, cy), (cx, cy + half)], fill=black, width=line_w)

    # 2. Joint dots – non-crossing cells only
    for y in range(height):
        for x in range(width):
            if (x, y) not in crossings:
                cx, cy = _center(x, y, cell_size)
                d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=black)

    # 3. Erase vertical pipe at each crossing (cut a white gap)
    for (x, y) in crossings:
        cx, cy = _center(x, y, cell_size)
        d.rectangle(
            [(cx - r - 2, cy - gap), (cx + r + 2, cy + gap)],
            fill=white,
        )

    # 4. Horizontal segments (drawn over the gap)
    for y in range(height):
        for x in range(width):
            cx, cy = _center(x, y, cell_size)
            cell = grid[y][x]
            if cell & E:
                d.line([(cx, cy), (cx + half, cy)], fill=black, width=line_w)
            if cell & W:
                d.line([(cx, cy), (cx - half, cy)], fill=black, width=line_w)

    # 5. Joint dots for crossing cells (on top)
    for (x, y) in crossings:
        cx, cy = _center(x, y, cell_size)
        d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=black)

    return img


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Pipe Maze Generator")

    with st.sidebar:
        width = st.number_input("Width (cells)", 5, 60, 15)
        height = st.number_input("Height (cells)", 5, 50, 15)
        cell_size = st.slider("Cell size (px)", 20, 100, 40)
        line_width = st.slider("Line thickness", 1, 20, 4)
        mode = st.radio("Mode", ["2D", "2.5D"])
        n_crossings = 0
        if mode == "2.5D":
            n_crossings = st.slider("Number of crossings", 0, 50, 12)
        generate = st.button("Generate")

    if generate or "pipe_maze" not in st.session_state:
        grid = generate_maze(int(width), int(height))
        crossings: set[tuple[int, int]] = set()
        if mode == "2.5D":
            crossings = add_crossings(grid, int(width), int(height), n_crossings)
        st.session_state["pipe_maze"] = (grid, crossings)

    grid, crossings = st.session_state["pipe_maze"]

    if mode == "2D":
        img = draw_2d(grid, int(cell_size), int(line_width))
    else:
        img = draw_25d(grid, crossings, int(cell_size), int(line_width))

    st.image(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button("Download PNG", buf.getvalue(), "pipe_maze.png", "image/png")


if __name__ == "__main__":
    main()
