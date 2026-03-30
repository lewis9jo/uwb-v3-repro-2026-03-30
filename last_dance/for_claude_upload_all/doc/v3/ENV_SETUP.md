# Environment setup for reproducible v3 experiment

이 폴더는 재현성 배포용 실험 코드 보관 구조입니다.

## 1) 파이썬 환경

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

권장:
- Python 3.10 이상
- GPU 사용 시 CUDA 호환 PyTorch 설치

## 2) 코드 실행 위치

- 실행 대상 스크립트: `v3_repro_all_in_one.py`
- 결과는 자동으로 `2026-03-30-v3/last_dance/0318_v3/outputs/` 하위에 저장

## 3) 데이터 경로

- raw 데이터: `new_bench_03/1_data/raw`
- processed 캐시: `new_bench_03/1_data/processed`  
- 재현용 번들 코드: `repro_bundle/new_bench_03` (실행 시 우선 사용)

## 4) 실행 예시

- 학습 + figure:
  - `python v3_repro_all_in_one.py --mode train`
  - `python v3_repro_all_in_one.py --mode train --tasks lab`
  - `python v3_repro_all_in_one.py --mode train --tasks classroom --classroom-baseline`

- figure 재생성:
  - `python v3_repro_all_in_one.py --mode figures`
  - `python v3_repro_all_in_one.py --mode figures --tasks lab --classroom-baseline`

## 5) GitHub 업로드 시 유의

- 코드와 실행 스크립트만 커밋
- `raw 데이터` / `캐시` / `체크포인트(.pth)` / `출력 그래프(.png, .pdf)`는 `.gitignore`로 제외 권장
- 큰 체크포인트는 필요 시 Git LFS(`*.pth`)로 분리

## 6) 재현 체크리스트 (커밋 전에 기록)

1. 코드 버전(커밋 해시)
2. `--seed`, 하이퍼파라미터(epochs/batch-size/lr 등)
3. 실행 모드(`train` 또는 `figures`)
4. 데이터셋 모드(`lab`, `classroom`, `all`)
5. 사용 데이터 버전/파일 경로
*** End Patch
