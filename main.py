"""
main.py
───────
26과학의날 미로 풀기 — 직렬 vs 병렬(MPI) A* 성능 비교.

사용법:
  # 직렬 실행 (베이스라인, MPI 불필요)
  python main.py

  # MPI 병렬 실행 (4 프로세스)
  mpirun -n 4 python main.py

  # 이미지 경로 지정
  python main.py mazes/maze.png
"""

import sys
import time
import numpy as np

# Windows CP949 콘솔에서 한글/특수문자 출력 보장
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from maze_reader import load_maze
from astar_serial import astar
from visualize import draw_path

# MPI 임포트 (런타임 없으면 직렬 전용 모드로 폴백)
try:
    from mpi4py import MPI
    from astar_mpi import hda_star, RANK, SIZE
    comm = MPI.COMM_WORLD
    MPI_AVAILABLE = True
except (ImportError, RuntimeError):
    MPI_AVAILABLE = False
    RANK = 0
    SIZE = 1


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "mazes/maze.png"

    if RANK == 0:
        print("=" * 60)
        print("  26과학의날 — A* 미로 탐색기")
        if not MPI_AVAILABLE:
            print("  [주의] MPI 런타임 미설치 — 직렬 모드로 실행합니다.")
        print("=" * 60)
        grid, start, goal = load_maze(image_path)
        print(f"  미로 크기  : {grid.shape[0]} x {grid.shape[1]} ({grid.size:,} 픽셀)")
        print(f"  시작 좌표  : {start}")
        print(f"  도착 좌표  : {goal}")
        print(f"  MPI 프로세스: {SIZE}")
        print("-" * 60)
    else:
        grid, start, goal = None, None, None

    if MPI_AVAILABLE:
        grid = comm.bcast(grid, root=0)
        start = comm.bcast(start, root=0)
        goal = comm.bcast(goal, root=0)

    # ══════════════════════════════════
    #  1) 직렬 A* (rank 0만 실행)
    # ══════════════════════════════════
    if RANK == 0:
        t0 = time.perf_counter()
        serial_path, serial_explored = astar(grid, start, goal)
        t1 = time.perf_counter()
        serial_time = t1 - t0

        if serial_path:
            print(f"  [직렬 A*]")
            print(f"    경로 길이  : {len(serial_path)} 칸")
            print(f"    탐색 노드  : {serial_explored:,} 개")
            print(f"    소요 시간  : {serial_time:.4f} 초")
        else:
            print("  [직렬 A*] 경로를 찾지 못했습니다.")
        print("-" * 60)

    if MPI_AVAILABLE:
        comm.Barrier()

        # ══════════════════════════════════
        #  2) 병렬 HDA* (모든 rank 참여)
        # ══════════════════════════════════
        t0 = MPI.Wtime()
        parallel_path, parallel_explored = hda_star(grid, start, goal)
        t1 = MPI.Wtime()
        parallel_time = t1 - t0

        if RANK == 0:
            print(f"  [병렬 HDA*] ({SIZE} 프로세스)")
            if parallel_path:
                print(f"    경로 길이  : {len(parallel_path)} 칸")
                print(f"    탐색 노드  : {parallel_explored:,} 개 (전체 합산)")
                print(f"    소요 시간  : {parallel_time:.4f} 초")
                speedup = serial_time / parallel_time if parallel_time > 0 else 0
                print(f"    속도 향상  : {speedup:.2f}x")
            else:
                print("    경로를 찾지 못했습니다.")
            print("=" * 60)

    if RANK == 0:
        best_path = (parallel_path if MPI_AVAILABLE else None) or serial_path
        if best_path:
            draw_path(image_path, best_path, output_path="result.png")
            print(f"\n  결과 이미지가 result.png 로 저장되었습니다.")


if __name__ == "__main__":
    main()
