import random
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

try:  # pragma: no cover - runtime dependency check
    from skimage import measure
except Exception:  # pragma: no cover - fallback install
    import sys
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp()
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--target",
            tmp,
            "scikit-image",
        ]
    )
    sys.path.append(tmp)
    from skimage import measure


def _extract_polygon(img: Image.Image, detail: float, smooth: float) -> np.ndarray:
    """Return a smoothed polygon outline from the given image."""
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
    return poly


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


def _generate_maze(grid_mask: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = grid_mask.shape
    visited = np.zeros_like(grid_mask)
    h_walls = np.ones((h + 1, w), dtype=bool)
    v_walls = np.ones((h, w + 1), dtype=bool)
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
    return h_walls, v_walls


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
    contour_pt: float,
    maze_pt: float,
    detail: float,
    smooth: float,
    scale: float,
    start: Tuple[int, int],
    end: Tuple[int, int],
):
    poly = _extract_polygon(img, detail, smooth)
    mask, full_mask = _masks_from_polygon(width, height, poly)
    if not mask[start] or not mask[end]:
        raise ValueError("Start lub meta poza konturem")
    h_walls, v_walls = _generate_maze(mask, start)
    path = _solve_maze(h_walls, v_walls, start, end, mask)
    w_img, h_img = poly[:, 0].max(), poly[:, 1].max()
    cell_size = min(w_img / width, h_img / height)
    w_svg = width * cell_size * scale
    h_svg = height * cell_size * scale
    scale_x = w_svg / w_img
    scale_y = h_svg / h_img
    poly_svg = [(x * scale_x, y * scale_y) for x, y in poly]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_svg}" height="{h_svg}" viewBox="0 0 {w_svg} {h_svg}">'
    ]
    path_d = "M " + " ".join(f"{x},{y}" for x, y in poly_svg) + " Z"
    svg.append(
        f'<path d="{path_d}" fill="none" stroke="black" stroke-width="{contour_pt}pt" />'
    )
    def inside_svg(x: float, y: float) -> bool:
        return _point_in_polygon(x, y, poly_svg)

    def _segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return t, ix, iy
        return None

    def _clip_segment(x1, y1, x2, y2):
        pts = [(0.0, x1, y1), (1.0, x2, y2)]
        n = len(poly_svg)
        for i in range(n):
            x3, y3 = poly_svg[i]
            x4, y4 = poly_svg[(i + 1) % n]
            inter = _segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            if inter:
                t, ix, iy = inter
                pts.append((t, ix, iy))
        pts.sort(key=lambda p: p[0])
        segs = []
        for a, b in zip(pts, pts[1:]):
            mx = (a[1] + b[1]) / 2
            my = (a[2] + b[2]) / 2
            if inside_svg(mx, my):
                segs.append((a[1], a[2], b[1], b[2]))
        return segs

    for r in range(height + 1):
        for c in range(width):
            if h_walls[r, c]:
                x1 = c * cell_size * scale
                y1 = r * cell_size * scale
                x2 = (c + 1) * cell_size * scale
                y2 = y1
                for sx1, sy1, sx2, sy2 in _clip_segment(x1, y1, x2, y2):
                    svg.append(
                        f'<line x1="{sx1}" y1="{sy1}" x2="{sx2}" y2="{sy2}" stroke="black" stroke-width="{maze_pt}pt" />'
                    )
    for r in range(height):
        for c in range(width + 1):
            if v_walls[r, c]:
                x1 = c * cell_size * scale
                y1 = r * cell_size * scale
                x2 = x1
                y2 = (r + 1) * cell_size * scale
                for sx1, sy1, sx2, sy2 in _clip_segment(x1, y1, x2, y2):
                    svg.append(
                        f'<line x1="{sx1}" y1="{sy1}" x2="{sx2}" y2="{sy2}" stroke="black" stroke-width="{maze_pt}pt" />'
                    )
    sx = start[1] * cell_size * scale + (cell_size * scale) / 2
    sy = start[0] * cell_size * scale + (cell_size * scale) / 2
    ex = end[1] * cell_size * scale + (cell_size * scale) / 2
    ey = end[0] * cell_size * scale + (cell_size * scale) / 2
    svg.append(f'<circle cx="{sx}" cy="{sy}" r="{cell_size * scale / 3}" fill="green" />')
    svg.append(f'<circle cx="{ex}" cy="{ey}" r="{cell_size * scale / 3}" fill="red" />')

    solution_lines = ""
    sol_segs = []
    pts = [
        (
            c * cell_size * scale + (cell_size * scale) / 2,
            r * cell_size * scale + (cell_size * scale) / 2,
        )
        for r, c in path
    ]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        sol_segs.extend(_clip_segment(x1, y1, x2, y2))
    if sol_segs:
        solution_lines = "".join(
            f'<line x1="{sx1}" y1="{sy1}" x2="{sx2}" y2="{sy2}" stroke="blue" stroke-width="{maze_pt}pt" />'
            for sx1, sy1, sx2, sy2 in sol_segs
        )

    def _open(poly_svg, cell):
        if full_mask[cell]:
            return
        cx = cell[1] * cell_size * scale + (cell_size * scale) / 2
        cy = cell[0] * cell_size * scale + (cell_size * scale) / 2
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            mx = cx + dc * (cell_size * scale / 2)
            my = cy + dr * (cell_size * scale / 2)
            if not inside_svg(mx, my):
                exx = cx + dc * cell_size * scale
                eyy = cy + dr * cell_size * scale
                svg.append(
                    f'<line x1="{cx}" y1="{cy}" x2="{exx}" y2="{eyy}" stroke="white" stroke-width="{contour_pt*2}pt" />'
                )
                break

    _open(poly_svg, start)
    _open(poly_svg, end)

    svg.append("</svg>")
    base_svg = "\n".join(svg)
    return base_svg, solution_lines, w_svg, h_svg


def main() -> None:
    st.title("Kontur Maze")
    uploaded = st.file_uploader("Wczytaj obraz", type=["png", "jpg", "jpeg"])
    width = st.sidebar.number_input("width", min_value=5, max_value=200, value=20)
    height = st.sidebar.number_input("height", min_value=5, max_value=200, value=20)
    contour_pt = st.sidebar.number_input("Grubość konturu (pt)", min_value=1.0, value=3.0)
    maze_pt = st.sidebar.number_input("Grubość labiryntu (pt)", min_value=0.5, value=1.0)
    detail = st.sidebar.slider("poziom detali", 1.0, 10.0, 2.0)
    smooth = st.sidebar.slider("wygładzenie", 0, 5, 0)
    scale = st.sidebar.slider("skala", 0.5, 5.0, 1.0)
    start_btn = st.sidebar.button("start")

    if uploaded:
        img = Image.open(uploaded)
        if start_btn or "outline" not in st.session_state:
            poly = _extract_polygon(img, detail, smooth)
            mask, full_mask = _masks_from_polygon(width, height, poly)
            w_img, h_img = poly[:, 0].max(), poly[:, 1].max()
            cell_size = min(w_img / width, h_img / height)
            w_svg = width * cell_size * scale
            h_svg = height * cell_size * scale
            scale_x = w_svg / w_img
            scale_y = h_svg / h_img
            poly_svg = [(x * scale_x, y * scale_y) for x, y in poly]
            st.session_state["outline"] = (
                poly_svg,
                mask,
                full_mask,
                cell_size,
                w_svg,
                h_svg,
                img,
            )
            st.session_state["start"] = None
            st.session_state["end"] = None
            st.session_state.pop("maze_svg", None)
            st.session_state.pop("solution_svg", None)

        if "outline" in st.session_state:
            (
                poly_svg,
                mask,
                full_mask,
                cell_size,
                w_svg,
                h_svg,
                img,
            ) = st.session_state["outline"]

            if st.session_state.get("maze_svg") is None:
                st.text_input("Kliknięta kratka", key="clicked_cell")
                rects: List[str] = []
                for r in range(height):
                    for c in range(width):
                        fill = "transparent"
                        if st.session_state.get("start") == (r, c):
                            fill = "#afa"
                        elif st.session_state.get("end") == (r, c):
                            fill = "#faa"
                        rects.append(
                            f'<rect data-r="{r}" data-c="{c}" x="{c * cell_size * scale}" y="{r * cell_size * scale}" width="{cell_size * scale}" height="{cell_size * scale}" fill="{fill}" stroke="#ddd" onclick="send(evt)" />'
                        )
                path_d = "M " + " ".join(f"{x},{y}" for x, y in poly_svg) + " Z"
                script = (
                    "<script>function send(evt){"  # noqa: E501
                    "const r=evt.target.getAttribute('data-r');"
                    "const c=evt.target.getAttribute('data-c');"
                    "const input=window.parent.document.querySelector('input[aria-label=\"Kliknięta kratka\"]');"
                    "if(input){input.value=r+','+c;input.dispatchEvent(new Event('input',{bubbles:true}));}}"  # noqa: E501
                    "</script>"
                )
                html = (
                    "<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>".format(
                        w=w_svg, h=h_svg
                    )
                    + "".join(rects)
                    + f'<path d="{path_d}" fill="none" stroke="black" stroke-width="{contour_pt}pt" />'
                    + "</svg>"
                    + script
                )
                components.html(html, height=int(h_svg) + 10)
                clicked = st.session_state.get("clicked_cell")
                rerun = False
                if clicked:
                    try:
                        r, c = map(int, clicked.split(","))
                        if st.session_state.get("start") is None:
                            st.session_state["start"] = (r, c)
                            rerun = True
                        elif st.session_state.get("end") is None and (r, c) != st.session_state.get(
                            "start"
                        ):
                            st.session_state["end"] = (r, c)
                            rerun = True
                    except Exception:
                        pass
                    finally:
                        st.session_state["clicked_cell"] = ""
                if rerun:
                    st.experimental_rerun()
                st.write("Start:", st.session_state.get("start"))
                st.write("End:", st.session_state.get("end"))

            if (
                st.session_state.get("start") is not None
                and st.session_state.get("end") is not None
                and st.session_state.get("maze_svg") is None
            ):
                maze_svg, sol_svg, w_svg, h_svg = generate_contour_maze(
                    img,
                    width,
                    height,
                    contour_pt,
                    maze_pt,
                    detail,
                    smooth,
                    scale,
                    st.session_state["start"],
                    st.session_state["end"],
                )
                st.session_state["maze_svg"] = maze_svg
                st.session_state["solution_svg"] = sol_svg
                st.session_state["w_svg"] = w_svg
                st.session_state["h_svg"] = h_svg
                st.experimental_rerun()

            if st.session_state.get("maze_svg"):
                if st.button("rozwiązanie"):
                    st.session_state["show_solution"] = not st.session_state.get(
                        "show_solution", False
                    )
                svg = st.session_state["maze_svg"]
                if st.session_state.get("show_solution") and st.session_state.get(
                    "solution_svg"
                ):
                    svg = svg.replace(
                        "</svg>", st.session_state["solution_svg"] + "</svg>"
                    )
                st.components.v1.html(svg, height=int(st.session_state["h_svg"] * 1.1))
                st.download_button(
                    "pobierz",
                    data=svg,
                    file_name="maze.svg",
                    mime="image/svg+xml",
                )


if __name__ == "__main__":
    main()
