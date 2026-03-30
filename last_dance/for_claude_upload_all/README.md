# v3 Repro Bundle (2026-03-30-v3)

이 폴더는 v3 재현 실행에 필요한 핵심 스크립트만 정리한 폴더입니다.

## 폴더 구성
- `last_dance/0318_v3/`
  - `run_v3_final.py`: v3 전체 파이프라인 실행 스크립트
  - `make_lab_figs.py`: lab CDF/t-SNE 생성
  - `make_classroom_figs.py`: classroom CDF/t-SNE 생성
  - `paper_v3_single_file.py`, `proposed_doc.py`: 실험 문서/요약용 보조 스크립트
  - `README.md`, `v3_file_summary.md`: 프로젝트 맥락 및 파일 요약
- `last_dance/v3_clean_run/`: 이전 clean run 클래스룸 그림 스크립트 (legacy)
- `last_dance/v3_lab_run/`: 이전 lab 그림 스크립트 (legacy)
- `last_dance/last_dance/v3_*_run/`: 기존 지표 CSV(비교용)

## 실행 예시
```bash
cd C:\Users\aarowin\Desktop\project
python last_dance\0318_v3\run_v3_final.py --tasks classroom --seed 44 --classroom-baseline
python last_dance\0318_v3\make_classroom_figs.py
python last_dance\0318_v3\make_lab_figs.py
```

참고: 현재 스크립트는 기존 `new_bench_03` 의존성을 사용합니다.
기존 체크포인트/원시 데이터 경로는 원본 위치를 그대로 참조하도록 작성되어 있어, 동일 저장소에서 실행해야 재현성이 가장 높습니다.
