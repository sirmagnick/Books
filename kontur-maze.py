import random
from typing import List, Tuple
from collections import deque

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

try:
    from skimage import measure
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scikit-image is required to compute image contours. "
        "Please add it to your environment's dependencies."
    ) from exc

Cell = Tuple[int, int]


def _carve_maze_mask(allowed: np.ndarray) -> Tuple[List[List[int]], Cell]:
    """Carve a maze confined to the allowed mask."""
    h, w = allowed.shape
    grid = [[1] * (w * 2 + 1) for _ in range(h * 2 + 1)]
    start: Cell = (0, 0)
    found = False
    for y in range(h):
        for x in range(w):
            if allowed[y, x]:
                start = (x, y)
                found = True
                break
        if found:
            break
    stack = [start]
    visited = {start}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        x, y = stack[-1]
        grid[y * 2 + 1][x * 2 + 1] = 0
        neighbors = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and allowed[ny, nx] and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            grid[y * 2 + 1 + dy][x * 2 + 1 + dx] = 0
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    return grid, start


def _solve_maze(grid: List[List[int]], start: Cell, allowed: np.ndarray) -> Tuple[Cell, List[Cell]]:
    """Find farthest reachable cell and path from start."""
    w = allowed.shape[1]
    h = allowed.shape[0]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < w
                and 0 <= ny < h
                and allowed[ny, nx]
                and grid[y * 2 + 1 + dy][x * 2 + 1 + dx] == 0
                and (nx, ny) not in dist
            ):
                dist[(nx, ny)] = dist[(x, y)] + 1
                queue.append((nx, ny))
    end = max(dist, key=dist.get)
    path = [end]
    while path[-1] != start:
        x, y = path[-1]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (
                (nx, ny) in dist
                and dist[(nx, ny)] == dist[(x, y)] - 1
                and grid[y * 2 + 1 + dy][x * 2 + 1 + dx] == 0
            ):
                path.append((nx, ny))
                break
    path.reverse()
    return end, path


def generate_contour_maze(
    src: Image.Image,
    w: int,
    h: int,
    scale: int,
    contour_pt: float,
    maze_pt: float,
    show_solution: bool,
) -> Tuple[str, int, int]:
    """Generate maze inside image contour and return SVG string."""
    allowed = np.array(src.convert("L").resize((w, h), Image.LANCZOS)) < 128
    grid, start = _carve_maze_mask(allowed)
    end, solution = _solve_maze(grid, start, allowed)

    width_px = w * scale
    height_px = h * scale
    mask_svg = (
        src.convert("L")
        .resize((width_px, height_px), Image.LANCZOS)
        .filter(ImageFilter.GaussianBlur(1))
    )
    mask_np = np.array(mask_svg) < 128
    contours = measure.find_contours(mask_np.astype(float), 0.5)
    contour = max(contours, key=len) if contours else []

    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">']
    if len(contour):
        pts = [(p[1], p[0]) for p in contour]
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts) + " Z"
        svg_parts.append(
            f'<path d="{d}" stroke="black" fill="none" stroke-width="{contour_pt}pt"/>'
        )

    cell = scale
    for y in range(h):
        for x in range(w):
            if not allowed[y, x]:
                continue
            if x + 1 < w and allowed[y, x + 1] and grid[y * 2 + 1][x * 2 + 2] == 1:
                x1 = (x + 1) * cell
                y1 = y * cell
                y2 = (y + 1) * cell
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x1}" y2="{y2}" stroke="black" stroke-width="{maze_pt}pt"/>'
                )
            if y + 1 < h and allowed[y + 1, x] and grid[y * 2 + 2][x * 2 + 1] == 1:
                y1 = (y + 1) * cell
                x1 = x * cell
                x2 = (x + 1) * cell
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y1}" stroke="black" stroke-width="{maze_pt}pt"/>'
                )

    def center(c: Cell) -> Tuple[float, float]:
        cx, cy = c
        return (cx + 0.5) * cell, (cy + 0.5) * cell

    sx, sy = center(start)
    ex, ey = center(end)
    svg_parts.append(
        f'<circle cx="{sx}" cy="{sy}" r="{cell * 0.2}" fill="green"/>'
    )
    svg_parts.append(
        f'<circle cx="{ex}" cy="{ey}" r="{cell * 0.2}" fill="red"/>'
    )

    if show_solution:
        pts = [center(p) for p in solution]
        pts_str = " ".join(f"{x},{y}" for x, y in pts)
        svg_parts.append(
            f'<polyline points="{pts_str}" stroke="blue" fill="none" stroke-width="{maze_pt}pt"/>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts), width_px, height_px


def main() -> None:
    st.title("kontur maze")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if "kontur_show_solution" not in st.session_state:
        st.session_state["kontur_show_solution"] = False
    with st.sidebar:
        w = st.number_input("width", 5, 100, 20)
        h = st.number_input("height", 5, 100, 20)
        scale = st.number_input("scale", 1, 50, 10)
        contour_thick = st.number_input("grubość konturu (pt)", 0.1, 20.0, 3.0)
        maze_thick = st.number_input("grubość labiryntu (pt)", 0.1, 10.0, 1.0)
        generate = st.button("Generate")
        if st.button("rozwiązanie"):
            st.session_state["kontur_show_solution"] = not st.session_state[
                "kontur_show_solution"
            ]
    show_solution = st.session_state["kontur_show_solution"]
    if generate:
        if uploaded is None:
            st.warning("Please upload an image.")
        else:
            src = Image.open(uploaded)
            svg, w_px, h_px = generate_contour_maze(
                src,
                int(w),
                int(h),
                int(scale),
                float(contour_thick),
                float(maze_thick),
                show_solution,
            )
            st.components.v1.html(svg, height=h_px + 10)
            st.download_button("Download SVG", svg, file_name="maze.svg")


if __name__ == "__main__":
    main()

