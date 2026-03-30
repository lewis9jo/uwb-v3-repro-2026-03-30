# 0318_v3 (v3 no-attention 정리)

기존 `v3_lab_run` / `v3_clean_run` 산재한 코드를 모아 정리한 폴더입니다.

## 위치
- `last_dance/0318_v3` (`C:\Users\aarowin\Desktop\project\last_dance\0318_v3`)

## 포함 코드
- `make_lab_figs.py`
  - 원본: `last_dance/v3_lab_run/make_lab_figs.py`
- `make_classroom_figs.py`
  - 원본: `last_dance/v3_clean_run/make_classroom_figs.py`
- `run_v3_final.py` (전체 파이프라인)
  - 데이터 로딩/전처리 → 학습 → 예측 → 평가 → Figure 저장까지 한 번에 실행
  - `--rebuild-data`: 원시 데이터 캐시 재생성
  - `--trace-data`: raw/filter/split 단계 로그 출력
  - `--classroom-baseline`: classroom CDF에서 LSTM/FC-SVM/EKF 기준선 오버레이
  - 하이퍼파라미터 오버라이드: `--seed`, `--batch-size`, `--epochs`, `--lr`, `--weight-decay` 등

## 산출물
- `outputs/lab/*`
- `outputs/classroom/*`
- `outputs/checkpoints/*`
- `outputs/pipeline_summary.csv`

## 기본 실행
- `python run_v3_final.py`
- `python run_v3_final.py --tasks lab`
- `python run_v3_final.py --tasks classroom --classroom-baseline --seed 44 --epochs 200 --lr 0.0008`
