# 26과학의날 — 병렬 A* 미로 탐색기

## 개요
[26science-day.pages.dev](https://26science-day.pages.dev)에서 협업으로 만든 미로 이미지를
**HDA* (Hash Distributed A*)** 알고리즘과 **mpi4py**를 이용해 병렬로 풀어주는 프로젝트입니다.

## 구조
```
maze_mpi/
├── main.py          # 메인 실행 (직렬 vs 병렬 비교)
├── maze_reader.py   # 이미지 → 이진 격자 변환
├── astar_serial.py  # 직렬 A* (베이스라인)
├── astar_mpi.py     # HDA* 병렬 A* (mpi4py)
├── visualize.py     # 결과 경로 시각화
└── README.md
```

## 설치
```bash
# 필수 패키지
pip install mpi4py numpy pillow

# MPI 런타임 (없으면 설치)
# Ubuntu/Debian
sudo apt install mpich
# macOS
brew install mpich
# Windows → Microsoft MPI 설치: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
```

## 사용법

### 1단계: 미로 이미지 준비
1. https://26science-day.pages.dev 접속
2. 각 칸에 미로 그리기 (초록=입구, 빨강=출구)
3. **"구역 경계선 포함" 체크 해제** 후 "전체 미로 병합해서 다운로드"
4. 다운로드한 PNG를 이 폴더에 `maze.png`로 저장

### 2단계: 실행
```bash
# 직렬 실행 (MPI 1프로세스 = 베이스라인과 동일)
python main.py maze.png

# 4 프로세스 병렬
mpirun -n 4 python main.py maze.png

# 8 프로세스 병렬
mpirun -n 8 python main.py maze.png
```

### 3단계: 결과 확인
- 콘솔에 직렬/병렬 소요시간, 속도향상 배율 출력
- `result.png`에 경로가 빨간색으로 표시된 이미지 저장

## 알고리즘: HDA* (Hash Distributed A*)

```
각 노드 (row, col) → hash → 담당 rank 결정
                             ↓
                    해당 rank의 로컬 open_list에 삽입
                             ↓
                    rank별 독립 탐색 (lock 없음!)
                             ↓
                    이웃이 다른 rank 소유 → MPI Send
```

### 왜 HDA*인가?
- 단순 공유 큐 방식은 lock 경합으로 병렬 효율이 떨어짐
- HDA*는 노드를 해시로 분배 → 각 rank가 완전히 독립적으로 동작
- 통신은 이웃 노드 전달 시에만 발생

### 종료 판정
- 모든 rank의 open_list 크기를 `MPI_Allreduce`로 합산
- 연속 5회 총합 0이면 종료 (in-flight 메시지 대응)

## 미로 이미지 규칙
| 색상 | 의미 | 격자 값 |
|------|------|---------|
| 검정 | 벽   | 1       |
| 흰색 | 통로 | 0       |
| 초록 | 입구 | 0 + start |
| 빨강 | 출구 | 0 + goal  |

## 성능 팁
- 미로 이미지가 클수록 (고해상도) 병렬 효과가 큼
- 프로세스 수는 CPU 코어 수에 맞추는 것이 최적
- 경계선 없이 다운로드해야 오탐 방지
