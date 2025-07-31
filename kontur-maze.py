import random
from typing import List, Tuple

import streamlit as st
from PIL import Image, ImageFilter

Cell = Tuple[int, int]


def _carve_maze(w: int, h: int) -> List[List[int]]:
    """Generate a rectangular maze using depth-first search."""
    grid = [[1] * (w * 2 + 1) for _ in range(h * 2 + 1)]
    stack = [(0, 0)]
    visited = {(0, 0)}
    while stack:
        x, y = stack[-1]
        grid[y * 2 + 1][x * 2 + 1] = 0
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            grid[y * 2 + 1 + dy][x * 2 + 1 + dx] = 0
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    return grid


def _grid_to_image(grid: List[List[int]]) -> Image.Image:
    """Convert grid representation into a PIL image."""
    h = len(grid)
    w = len(grid[0])
    img = Image.new("1", (w, h), 1)
    px = img.load()
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val == 1:
                px[x, y] = 0
    return img


def generate_contour_maze(src: Image.Image, w: int, h: int, scale: int, thickness: int) -> Image.Image:
    """Draw the outline of ``src`` and generate a maze inside it."""
    grid = _carve_maze(w, h)
    maze_img = _grid_to_image(grid)

    mask = src.convert("L").resize(maze_img.size)
    mask = mask.point(lambda p: 255 if p < 128 else 0, mode="1")

    edge = mask.filter(ImageFilter.FIND_EDGES)
    size = max(1, int(thickness))
    if size % 2 == 0:
        size += 1
    edge = edge.filter(ImageFilter.MaxFilter(size)).convert("1")

    base = Image.new("1", maze_img.size, 1)
    base.paste(maze_img, mask=mask)
    base.paste(0, mask=edge)

    if scale > 1:
        base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    return base


def main() -> None:
    st.title("kontur maze")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    with st.sidebar:
        w = st.number_input("width", 5, 100, 20)
        h = st.number_input("height", 5, 100, 20)
        scale = st.number_input("scale", 1, 50, 10)
        thick = st.number_input("contour thickness", 1, 20, 3)
        generate = st.button("Generate")
    if generate:
        if uploaded is None:
            st.warning("Please upload an image.")
        else:
            src = Image.open(uploaded)
            img = generate_contour_maze(src, int(w), int(h), int(scale), int(thick))
            st.image(img)


if __name__ == "__main__":
    main()
