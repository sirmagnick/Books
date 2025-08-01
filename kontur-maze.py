import random
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st

try:  # pragma: no cover - runtime dependency check
    from skimage import measure
except Exception:
    import sys
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", tmp, "scikit-image"]
    )
    sys.path.append(tmp)
    from skimage import measure


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    px = [p[0] for p in poly]
    py = [p[1] for p in poly]
    j = n - 1
    for i in range(n):
        if ((py[i] > y) != (py[j] > y)) and (
            x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        ):
            inside = not inside
        j = i
    return inside


def _masks_from_polygon(
    width: int, height: int, poly: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    grid_poly: List[Tuple[float, float]] = []
    for x, y in poly:
        gx = x / poly[:, 0].max() * width
        gy = y / poly[:, 1].max() * height
        grid_poly.append((gx, gy))
    center_mask = np.zeros((height, width), dtype=bool)
    full_mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            cx, cy = c + 0.5, r + 0.5
            if _point_in_polygon(cx, cy, grid_poly):
                center_mask[r, c] = True
            corners = [(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)]
            if all(_point_in_polygon(x, y, grid_poly) for x, y in corners):
                full_mask[r, c] = True
    return center_mask, full_mask


def _smooth_polygon(poly: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return poly
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    for _ in range(iterations):
        new_pts = []
        for i in range(len(poly) - 1):
            p0, p1 = poly[i], poly[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        poly = np.array(new_pts + [new_pts[0]])
    return poly


def _generate_maze(grid_mask: np.ndarray):
    h, w = grid_mask.shape
    visited = np.zeros_like(grid_mask)
    h_walls = np.ones((h + 1, w), dtype=bool)
    v_walls = np.ones((h, w + 1), dtype=bool)
    start = tuple(np.argwhere(grid_mask)[0])
    stack = [start]
    visited[start] = True
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        r, c = stack[-1]
        neigh = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and grid_mask[nr, nc]
                and not visited[nr, nc]
            ):
                neigh.append((nr, nc, dr, dc))
        if neigh:
            nr, nc, dr, dc = random.choice(neigh)
            if dr == 1:
                h_walls[r + 1, c] = False
            if dr == -1:
                h_walls[r, c] = False
            if dc == 1:
                v_walls[r, c + 1] = False
            if dc == -1:
                v_walls[r, c] = False
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    inside = np.argwhere(grid_mask)
    start = tuple(inside[0])
    end = tuple(inside[-1])
    return h_walls, v_walls, start, end


def _solve_maze(h_walls, v_walls, start, end, mask):
    h, w = mask.shape
    q = deque([start])
    prev = {start: None}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                continue
            if (nr, nc) in prev:
                continue
            if dr == 1 and h_walls[r + 1, c]:
                continue
            if dr == -1 and h_walls[r, c]:
                continue
            if dc == 1 and v_walls[r, c + 1]:
                continue
            if dc == -1 and v_walls[r, c]:
                continue
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    return path[::-1]


def generate_contour_maze(
    img: Image.Image,
    width: int,
    height: int,
    cell_size: int,
    contour_pt: float,
    maze_pt: float,
    detail: float,
    smooth: float,
    scale: float,
    show_solution: bool,
):
    gray = img.convert("L")
    arr = np.array(gray)
    arr = arr < 128
    contours = measure.find_contours(arr.astype(float), 0.5)
    if not contours:
        raise ValueError("Nie znaleziono konturu")
    contour = max(contours, key=len)
    poly = measure.approximate_polygon(contour, tolerance=detail)
    poly = _smooth_polygon(poly, int(smooth))
    poly -= poly.min(axis=0)
    mask, full_mask = _masks_from_polygon(width, height, poly)
    h_walls, v_walls, _, _ = _generate_maze(mask)
    inside_full = np.argwhere(full_mask)
    if len(inside_full) < 2:
        raise ValueError("Kontur zbyt mały dla labiryntu")
    start = tuple(inside_full[0])
    end = tuple(inside_full[-1])
    path = _solve_maze(h_walls, v_walls, start, end, full_mask)
    w_svg = width * cell_size * scale
    h_svg = height * cell_size * scale
    poly_svg = []
    for x, y in poly:
        sx = x / poly[:, 0].max() * w_svg
        sy = y / poly[:, 1].max() * h_svg
        poly_svg.append((sx, sy))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_svg}" height="{h_svg}" viewBox="0 0 {w_svg} {h_svg}">'
    ]
    path_d = "M " + " ".join(f"{x},{y}" for x, y in poly_svg) + " Z"
    svg.append(
        f'<path d="{path_d}" fill="none" stroke="black" stroke-width="{contour_pt}pt" />'
    )
    def inside_svg(x: float, y: float) -> bool:
        return _point_in_polygon(x, y, poly_svg)

    for r in range(height + 1):
        for c in range(width):
            if h_walls[r, c]:
                x1 = c * cell_size * scale
                y1 = r * cell_size * scale
                x2 = (c + 1) * cell_size * scale
                y2 = y1
                if inside_svg(x1, y1) and inside_svg(x2, y2):
                    svg.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{maze_pt}pt" />'
                    )
    for r in range(height):
        for c in range(width + 1):
            if v_walls[r, c]:
                x1 = c * cell_size * scale
                y1 = r * cell_size * scale
                x2 = x1
                y2 = (r + 1) * cell_size * scale
                if inside_svg(x1, y1) and inside_svg(x2, y2):
                    svg.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{maze_pt}pt" />'
                    )
    sx = start[1] * cell_size * scale + (cell_size * scale) / 2
    sy = start[0] * cell_size * scale + (cell_size * scale) / 2
    ex = end[1] * cell_size * scale + (cell_size * scale) / 2
    ey = end[0] * cell_size * scale + (cell_size * scale) / 2
    svg.append(f'<circle cx="{sx}" cy="{sy}" r="{cell_size * scale / 3}" fill="green" />')
    svg.append(f'<circle cx="{ex}" cy="{ey}" r="{cell_size * scale / 3}" fill="red" />')
    if show_solution:
        pts = []
        for r, c in path:
            x = c * cell_size * scale + (cell_size * scale) / 2
            y = r * cell_size * scale + (cell_size * scale) / 2
            pts.append((x, y))
        filt = [pts[0]]
        for i in range(1, len(pts)):
            mx = (pts[i - 1][0] + pts[i][0]) / 2
            my = (pts[i - 1][1] + pts[i][1]) / 2
            if inside_svg(mx, my):
                filt.append(pts[i])
        pts_str = " ".join(f"{x},{y}" for x, y in filt)
        svg.append(
            f'<polyline points="{pts_str}" fill="none" stroke="blue" stroke-width="{maze_pt}pt" />'
        )
    svg.append("</svg>")
    return "\n".join(svg), w_svg, h_svg


def main() -> None:
    st.title("Kontur Maze")
    uploaded = st.file_uploader("Wczytaj obraz", type=["png", "jpg", "jpeg"])
    width = st.sidebar.number_input("width", min_value=5, max_value=200, value=20)
    height = st.sidebar.number_input("height", min_value=5, max_value=200, value=20)
    cell = st.sidebar.number_input("cell size", min_value=5, max_value=100, value=20)
    contour_pt = st.sidebar.number_input("Grubość konturu (pt)", min_value=1.0, value=3.0)
    maze_pt = st.sidebar.number_input("Grubość labiryntu (pt)", min_value=0.5, value=1.0)
    detail = st.sidebar.slider("poziom detali", 1.0, 10.0, 2.0)
    smooth = st.sidebar.slider("wygładzenie", 0, 5, 0)
    scale = st.sidebar.slider("skala", 0.5, 5.0, 1.0)
    show_solution = st.checkbox("rozwiązanie")
    generate = st.button("generuj")
    if generate:
        st.session_state["kontur_run"] = True
    if uploaded and st.session_state.get("kontur_run"):
        img = Image.open(uploaded)
        svg, w, h = generate_contour_maze(
            img,
            width,
            height,
            cell,
            contour_pt,
            maze_pt,
            detail,
            smooth,
            scale,
            show_solution,
        )
        st.components.v1.html(svg, height=int(h * 1.1))
        st.download_button(
            "pobierz", data=svg, file_name="maze.svg", mime="image/svg+xml"
        )


if __name__ == "__main__":
    main()
