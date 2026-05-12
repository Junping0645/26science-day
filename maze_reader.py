"""
maze_reader.py
──────────────
미로 이미지를 로드하여 이진 격자(grid)로 변환합니다.

색상 규칙 (26science-day 기준):
  - 검정(벽)          → grid = 1
  - 흰색(통로)        → grid = 0
  - 초록(연결 통로)   → grid = 0  (구역 간 연결점 — 통로로 유지)
  - 빨강(구역 경계선) → grid = 1  (벽으로 처리)
"""

import numpy as np
from PIL import Image


def load_maze(image_path: str, wall_threshold: int = 100):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.int16)

    r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # ── 색상 판별 ──
    # 빨강: 구역 경계선 → 벽
    red_mask   = (r_ch - g_ch > 80) & (r_ch - b_ch > 80) & (r_ch > 150)
    # 초록: 구역 간 연결 통로 → 통로 유지
    green_mask = (g_ch - r_ch > 50) & (g_ch - b_ch > 50) & (g_ch > 100)

    # ── 벽/통로 이진화 ──
    gray = np.mean(arr, axis=2)
    grid = (gray < wall_threshold).astype(np.uint8)

    # 빨강 경계선 → 벽
    grid[red_mask] = 1
    # 초록 연결점 → 통로 (검정 판정 덮어쓰기)
    grid[green_mask] = 0

    # ── 시작/도착 자동 탐지 ──
    start = _best_edge_passage(grid, "left")
    goal  = _best_edge_passage(grid, "right")

    print(f"[미로] 시작: {start}  도착: {goal}")
    return grid, start, goal


def _best_edge_passage(grid: np.ndarray, edge: str):
    rows, cols = grid.shape
    if edge == "left":
        col = 0
        runs = _find_runs([r for r in range(rows) if grid[r, col] == 0])
        if not runs:
            return (0, 0)
        best = max(runs, key=lambda x: x[1])
        return (best[0] + best[1] // 2, col)
    if edge == "right":
        col = cols - 1
        runs = _find_runs([r for r in range(rows) if grid[r, col] == 0])
        if not runs:
            return (0, col)
        best = max(runs, key=lambda x: x[1])
        return (best[0] + best[1] // 2, col)
    return (0, 0)


def _find_runs(positions: list):
    if not positions:
        return []
    runs, start, length = [], positions[0], 1
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1] + 1:
            length += 1
        else:
            runs.append((start, length))
            start, length = positions[i], 1
    runs.append((start, length))
    return runs


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "mazes/maze.png"
    grid, start, goal = load_maze(path)
    print(f"미로 크기: {grid.shape[0]} x {grid.shape[1]}")
    print(f"벽 비율:   {grid.sum() / grid.size * 100:.1f}%")
