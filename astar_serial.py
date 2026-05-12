"""
astar_serial.py
───────────────
단일 프로세스 A* 탐색 (속도 비교용 베이스라인).
"""

import heapq
import numpy as np


def heuristic(a: tuple, b: tuple) -> int:
    """맨해튼 거리"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: np.ndarray, start: tuple, goal: tuple):
    """
    A* 최단 경로 탐색.

    Returns
    -------
    path : list[tuple] | None
        (row, col) 리스트. 경로 없으면 None.
    explored : int
        탐색한 노드 수.
    """
    rows, cols = grid.shape
    open_heap = [(heuristic(start, goal), 0, start)]  # (f, g, node)
    g_score = {start: 0}
    came_from = {}
    explored = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            return _reconstruct(came_from, start, goal), explored

        if g > g_score.get(current, float("inf")):
            continue  # stale

        explored += 1

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                new_g = g + 1
                neighbor = (nr, nc)
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    came_from[neighbor] = current
                    heapq.heappush(
                        open_heap,
                        (new_g + heuristic(neighbor, goal), new_g, neighbor),
                    )

    return None, explored  # 경로 없음


def _reconstruct(came_from, start, goal):
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    return path[::-1]