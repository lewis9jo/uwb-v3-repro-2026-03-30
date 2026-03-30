# v3 모델 관련 파일 정리 (for paper)

기준일: 2026-03-18  
폴더 기준: `c:\Users\aarowin\Desktop\project\last_dance\0318_v3`

## 0) 논문
- `c:\Users\aarowin\Desktop\project\Multi_Task_Learning_for_Joint_UWB_Positioning_and_NLOS_Classification-2.pdf`
  - 대상 논문 본문(PDF)

## 1) v3 메인 실험(권장 실행 경로)
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\README.md`
  - v3 폴더 설명, 실행 예시, 파일 매핑
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\run_v3_final.py`
  - v3 전체 파이프라인 실행 스크립트
  - 데이터 로드→학습→추론→지표→결과 저장→CDF/t-SNE 생성
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\paper_v3_single_file.py`
  - 논문 제출용 단일 파일형 파이프라인
  - 데이터/학습/평가/요약 CSV만 집중
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\proposed_doc.py`
  - v3 No-Attention 흐름을 가독성 있게 정리한 버전

## 2) 결과(그래프) 재생성 코드
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\make_lab_figs.py`
  - Lab 데이터셋용 CDF/ t-SNE 생성
  - 입력: lab test split, 체크포인트
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\make_classroom_figs.py`
  - Classroom 데이터셋용 CDF/t-SNE 생성
  - baseline( LSTM/FC-SVM/EKF ) 오버레이 지원

## 3) v3 핵심 산출물
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\metrics\v3_lab_metrics.csv`
  - lab 실험 지표
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\metrics\v3_classroom_metrics.csv`
  - classroom 실험 지표
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\lab\v3_lab_errors_cm.npy`
  - lab 오차 배열(CM)
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\classroom\v3_classroom_errors_cm.npy`
  - classroom 오차 배열(CM)
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\classroom\v3_classroom_features.npy`
  - 공유 임베딩(트렁크 출력) 배열
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\classroom\v3_classroom_labels.npy`
  - classroom 라벨 배열
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\lab\v3_lab_cdf.pdf`
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\classroom\v3_classroom_cdf.pdf`
  - CDF 플롯(PDF)
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\lab\v3_lab_tsne.pdf`
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\classroom\v3_classroom_tsne.pdf`
  - t-SNE 플롯(PDF)
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\errors\v3_lab_seed44_errors.npy`
- `c:\Users\aarowin\Desktop\project\last_dance\0318_v3\outputs\errors\v3_classroom_seed44_errors.npy`
  - seed 44 오차 캐시

## 4) v3 파생 산출 디렉터리(과거/비교실험)
- `c:\Users\aarowin\Desktop\project\last_dance\v3_clean_run`
  - classroom 전용 산출물(예: `v3_classroom_cdf.pdf`, `plots/*`, seed별 에러)
- `c:\Users\aarowin\Desktop\project\last_dance\v3_lab_run`
  - lab 전용 산출물(예: `v3_lab_cdf.pdf`, `v3_lab_tsne.pdf`, seed별 에러)
- `c:\Users\aarowin\Desktop\project\last_dance\last_dance\v3_clean_run`
  - 메인 clean run 버전(교차 실험용)
- `c:\Users\aarowin\Desktop\project\last_dance\last_dance\v3_lab_run`
  - 메인 lab run 버전(교차 실험용)
- `c:\Users\aarowin\Desktop\project\last_dance\last_dance\v3_clean_ablation`
  - ablation(무주의/구성요소 제거) 실험
- `c:\Users\aarowin\Desktop\project\last_dance\last_dance\v3_lab_conf_only`
  - confidence-only 변형 실험

## 5) v3 확인용 추천 체크 플로우
1. 논문/방법 확인: `Multi_Task_Learning_for_Joint_UWB_Positioning_and_NLOS_Classification-2.pdf`
2. 재현 실행: `paper_v3_single_file.py` 또는 `run_v3_final.py`
3. 결과 확인:
   - 지표: `outputs/metrics/*.csv`
   - 그래프: `outputs/*/*.pdf`
4. figure 재생성 비교: `make_lab_figs.py`, `make_classroom_figs.py`
