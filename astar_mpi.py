"""
astar_mpi.py
────────────
HDA* (Hash Distributed A*) — mpi4py 기반 병렬 A* 탐색.

종료 조건: 송신 총합 == 수신 총합 AND 전체 힙 크기 == 0
→ 비행 중인 메시지가 없음을 보장.
"""

from mpi4py import MPI
import heapq
import numpy as np

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

TAG_NODE = 0
TAG_GOAL = 1


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


_COL_WIDTH = None  # hda_star 호출 시 초기화

def node_owner(node):
    # 열(column) 기반 분할: 같은 구역의 노드를 같은 rank에 묶어 통신 최소화
    return min(node[1] // _COL_WIDTH, SIZE - 1)


def hda_star(grid: np.ndarray, start: tuple, goal: tuple):
    global _COL_WIDTH
    rows, cols = grid.shape
    _COL_WIDTH = max(1, cols // SIZE)

    open_heap = []
    g_score   = {}
    came_from = {}
    local_explored = 0
    best_cost = float("inf")

    total_sent = 0   # 이 rank가 보낸 TAG_NODE 수
    total_recv = 0   # 이 rank가 받은 TAG_NODE 수

    if node_owner(start) == RANK:
        g_score[start] = 0
        heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    while True:

        # ── 1) 수신 메시지 전부 처리 ──
        status = MPI.Status()
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_NODE, status=status):
            node, new_g, parent = comm.recv(source=status.Get_source(), tag=TAG_NODE)
            total_recv += 1
            if new_g < g_score.get(node, float("inf")):
                g_score[node]   = new_g
                came_from[node] = parent
                heapq.heappush(open_heap,
                               (new_g + heuristic(node, goal), new_g, node))

        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_GOAL, status=status):
            cost = comm.recv(source=status.Get_source(), tag=TAG_GOAL)
            best_cost = min(best_cost, cost)

        # ── 2) 노드 확장 ──
        for _ in range(min(len(open_heap), 50)):
            if not open_heap:
                break

            _, g, current = heapq.heappop(open_heap)

            if g > g_score.get(current, float("inf")):
                continue
            if g >= best_cost:
                continue

            if current == goal:
                if g < best_cost:
                    best_cost = g
                    for r in range(SIZE):
                        if r != RANK:
                            comm.send(g, dest=r, tag=TAG_GOAL)
                continue

            local_explored += 1

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr, nc] != 0:
                    continue
                neighbor = (nr, nc)
                new_g = g + 1
                if new_g >= best_cost:
                    continue

                owner = node_owner(neighbor)
                if owner == RANK:
                    if new_g < g_score.get(neighbor, float("inf")):
                        g_score[neighbor]   = new_g
                        came_from[neighbor] = current
                        heapq.heappush(open_heap,
                                       (new_g + heuristic(neighbor, goal),
                                        new_g, neighbor))
                else:
                    total_sent += 1
                    comm.send((neighbor, new_g, current), dest=owner, tag=TAG_NODE)

        # ── 3) 종료 판정: 힙 크기 + 송수신 균형 ──
        local_heap  = len(open_heap)
        global_heap = comm.allreduce(local_heap,  op=MPI.SUM)
        global_sent = comm.allreduce(total_sent,  op=MPI.SUM)
        global_recv = comm.allreduce(total_recv,  op=MPI.SUM)

        # 모든 힙이 비어있고 비행 중인 메시지가 없을 때만 종료
        if global_heap == 0 and global_sent == global_recv:
            break

    # ── 4) 잔류 메시지 최종 수거 ──
    status = MPI.Status()
    while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_NODE, status=status):
        node, new_g, parent = comm.recv(source=status.Get_source(), tag=TAG_NODE)
        if new_g < g_score.get(node, float("inf")):
            g_score[node]   = new_g
            came_from[node] = parent
    while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_GOAL, status=status):
        cost = comm.recv(source=status.Get_source(), tag=TAG_GOAL)
        best_cost = min(best_cost, cost)

    # ── 5) came_from 수집 → rank 0에서 경로 복원 ──
    all_came_from  = comm.gather(came_from,      root=0)
    total_explored = comm.reduce(local_explored, op=MPI.SUM, root=0)

    if RANK != 0:
        return None, 0

    merged = {}
    for d in all_came_from:
        merged.update(d)

    if goal not in merged and start != goal:
        return None, total_explored

    path, cur, visited = [], goal, set()
    while cur != start:
        if cur in visited or cur not in merged:
            return None, total_explored
        visited.add(cur)
        path.append(cur)
        cur = merged[cur]
    path.append(start)
    return path[::-1], total_explored
