# 2026-03-30-v3 모델 업로드 요약 (v3 UWB Multi-task)

이 문서는 `2026-03-30-v3` 폴더의 실험 코드 및 모델을 설명하기 위한 요약본입니다.  
목적은 ChatGPT/Claude가 코드를 보고 바로 모델 구조, 학습 파이프라인, 이론적 가정, 실험 재현 과정을 이해할 수 있게 만드는 것입니다.

## 1) 모델 정체성 (What this project is)

- 대상: UWB 기반 위치 추정 + NLOS 분류를 **동시 학습**하는 멀티태스크 신경망.
- 핵심 스크립트: [run_v3_final.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\run_v3_final.py)
- 실험 목표:
  - 레이블(LOS/Static NLOS/Dynamic NLOS) 예측 성능
  - XY 위치 오차를 중심으로한 회귀 성능(CDF/CEP50/CEP95)
  - 두 과제를 공유 표현(shared representation)로 학습해 서로 보완

## 2) 실험 파이프라인

`run_v3_final.py`에서 아래 순서로 동작합니다.

1) 데이터 적재/정제/분할
2) 피쳐 프로파일 선택 (base 또는 변형)
3) 모델 생성 (기본: `Proposed_V2_NoAttention`)
4) 학습 (Train / Val) + Early stopping
5) Test 추론 및 메트릭 계산
6) 오차 저장, 체크포인트 복사
7) CDF + t-SNE 저장

요약 출력은 `pipeline_summary.csv`에 누적됩니다.

## 3) 데이터 처리 이론(핵심)

- 데이터 공급자: [provider.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\repro_bundle\new_bench_03\data\provider.py)
- 원시 데이터: 엑셀(`*.xlsx`) + `new_bench_03/1_data/raw`
- 라벨 정합성: 환경 코드 재매핑 `{3->1, 4->2}`로 LOS/StaticNLOS/DynamicNLOS를 0/1/2로 통일
- 분할 전략:
  - 각 point_id, nlos_type 그룹 안에서 시계열 순서로 60% train, 20% val, 나머지 test
  - 매우 작은 샘플은 최소 1개를 train으로 보장
- 결측/특성 처리:
  - anchor별/point별 품질 필터(combo):
    - SNR, CIR 필터 + 이상치(Z-score) 제거
    - SNR 가중 이동평균으로 거리(DIST) 보정
  - 표준화(`StandardScaler`)는 train에서 fit 후 전체 split에 적용
  - 결측은 열 평균으로 치환

## 4) 피쳐 구성(중요)

[provider.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\repro_bundle\new_bench_03\data\provider.py)에서 각 앵커 샘플당 길이 20 벡터를 생성합니다.

- main(현재 앵커): `DIST, RSSI, SEQ, LOSS, CIR, FP2, MaxNoise, SNR` (8개)
- other(다른 3개 앵커 각 4개씩): `DIST, RSSI, CIR, SNR` (총 12개)
- 총합 20차원
- 좌표 라벨: `[x_m, y_m]`
- 라벨: `nlos_type` (정수 클래스)

### 피쳐 프로파일(변형)

- 구현: [adapters.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\repro_bundle\new_bench_03\data\adapters.py)
- 제공 프로파일 예: `base`, `low_dim`, `mid_dim`, `high_dim`, `base_no_snr`, `base_8d` 등
- `base_no_snr`: SNR 위치(7,11,15,19)를 0으로 강제
- `base_8d`: 8차원 이후를 0으로 차단
- 기본 실행은 `--profile base`(기본값)으로 동작

## 5) 모델 수식/구조(멀티태스크)

핵심 모델: [multitask_attention.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\repro_bundle\new_bench_03\models\proposed\multitask_attention.py)

- 입력을 main/other 두 브랜치로 분리(8,12)
- 각 브랜치: MLP(160→80→40)
- 옵션:
  - attention ON이면 브랜치 2개 가중치 조합
  - NO-ATTENTION variant는 단순 concat(main, other)
- 공통 shared trunk: `80→640→320→160`
- 헤드:
  - position head: shared → 2 (x,y)
  - class head: shared → num_classes
  - confidence head(선택): shared → sigmoid 1차원

### 손실 함수

- `UWBMultiTaskLoss`
  - 위치오차: `0.6 * SmoothL1 + 0.4 * MSE`
  - NLOS 보정(옵션): NLOS 샘플이면 위치 오차 가중치 상승
  - 분류: CrossEntropy(클래스 불균형 가중치 적용 가능)
  - confidence(옵션): BCE, target = `1 - clamp(position_error / conf_norm, 0, 1)`
  - 총합: `pos_weight*pos_loss + cls_weight*cls_loss + conf_weight*conf_loss`

## 6) 학습/추론 하이퍼파라미터

- 기본 실행은 `MODEL_NAME = Proposed_V2_NoAttention`
- `run_v3_final.py` 기본값:
  - seed 44
  - batch-size 128
  - epochs 200, min_epochs 60, patience 45
  - lr 8e-4, weight_decay 2e-4
  - cls_weight 2.5, conf_weight 0.7, pos_weight 3.0, nlos_boost 0.5
- 최적 모델 저장: `proposed_best_c{num_classes}.pth`
- 체크포인트 복사본: `outputs/lab|classroom/v3_*_seed{seed}_best.pth`

## 7) 평가 지표

- 위치: MAE, RMSE, CEP50, CEP95 (오차 단위는 cm)
- 분류: Accuracy, F1-macro, Precision/Recall macro, confusion matrix, class별 MAE/Accuracy
- 구현: [evaluator.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\repro_bundle\new_bench_03\output\evaluator.py)

## 8) 결과 산출물

- `outputs/pipeline_summary.csv`
- `outputs/lab/*`, `outputs/classroom/*`:
  - `v3_*_errors_cm.npy`
  - `v3_*_cdf.pdf/png`, `v3_*_tsne.pdf/png`
  - classroom의 경우 features/labels (`*_features.npy`, `*_labels.npy`)
- `outputs/checkpoints/.../proposed_best_c2_or_c3.pth`

## 9) 실험 스크립트 맵

- 실행 파이프라인: [run_v3_final.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\run_v3_final.py)
- 단일 파일 실험 흐름(이론/논문형): [paper_v3_single_file.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\paper_v3_single_file.py)
- 코드형 설명형 버전: [proposed_doc.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\proposed_doc.py)
- Figure 생성:
  - lab: [make_lab_figs.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\make_lab_figs.py)
  - classroom: [make_classroom_figs.py](c:\Users\aarowin\Desktop\project\2026-03-30-v3\last_dance\0318_v3\make_classroom_figs.py)

## 10) 실행 예시

```bash
python last_dance\0318_v3\run_v3_final.py --tasks classroom --seed 44 --classroom-baseline
python last_dance\0318_v3\run_v3_final.py --tasks lab --seed 44
python last_dance\0318_v3\make_classroom_figs.py
python last_dance\0318_v3\make_lab_figs.py
```

## 11) 업로드 전 핵심 메시지(모델 설명용 한줄 요약)

- 이 프로젝트는 “멀티태스크 UWB 위치추정-분류 통합 모델”로, 앵커간 상대 신호 특성으로 만든 20차원 피쳐를 사용해 위치와 LOS/NLOS를 동시에 예측하고, 위치 오차는 매끄러운 회귀손실+MSE로 학습하며 분류/신뢰도 항목과 결합해 최적화합니다.
- `Proposed_V2_NoAttention`은 실험의 기본 모델이며, attention을 제외한 variant이지만 동일한 멀티헤드 구조/학습 체계를 유지합니다.
