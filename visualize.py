"""
visualize.py
────────────
A* 탐색 결과 경로를 미로 이미지 위에 빨간색으로 그려서 저장.
"""

import numpy as np
from PIL import Image, ImageDraw


def draw_path(
    image_path: str,
    path: list,
    output_path: str = "result.png",
    line_width: int = 2,
    color: tuple = (255, 40, 40),
):
    """
    원본 미로 이미지 위에 경로를 빨간색으로 그려서 저장.

    Parameters
    ----------
    image_path  : 원본 미로 이미지 경로
    path        : [(row, col), ...] 경로 좌표 리스트
    output_path : 결과 저장 경로
    line_width  : 경로 선 두께
    color       : 경로 색상 (R, G, B)
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    # 경로 픽셀에 색칠
    for r, c in path:
        # line_width 만큼 주변 픽셀도 같이 색칠
        hw = line_width // 2
        r_start = max(0, r - hw)
        r_end = min(arr.shape[0], r + hw + 1)
        c_start = max(0, c - hw)
        c_end = min(arr.shape[1], c + hw + 1)
        arr[r_start:r_end, c_start:c_end] = color

    # 시작점 (초록) / 도착점 (파랑) 표시
    if path:
        _draw_marker(arr, path[0], (0, 200, 0), radius=max(3, line_width * 2))
        _draw_marker(arr, path[-1], (0, 80, 255), radius=max(3, line_width * 2))

    result = Image.fromarray(arr)
    result.save(output_path)
    print(f"[저장] 결과 이미지 → {output_path}")


def _draw_marker(arr, pos, color, radius=5):
    """원형 마커를 그립니다."""
    r, c = pos
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                nr, nc = r + dr, c + dc
                if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
                    arr[nr, nc] = color