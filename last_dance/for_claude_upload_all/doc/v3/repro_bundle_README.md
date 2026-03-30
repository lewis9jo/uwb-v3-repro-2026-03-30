# Reproducible code bundle

이 폴더에는 `v3_repro_all_in_one.py`에서 직접 참조되는 실험 공통 코드의 핵심 모듈을
하나로 묶어 복사해 둔 것입니다.

- `new_bench_03/__init__.py`
- `new_bench_03/paths.py`
- `new_bench_03/data/{__init__.py, dataset.py, provider.py, adapters.py, filters.py}`
- `new_bench_03/models/{__init__.py, base.py, registry.py}`
- `new_bench_03/models/proposed/{__init__.py, multitask_attention.py, variants.py}`
- `new_bench_03/output/{__init__.py, evaluator.py}`

실행 시 `v3_repro_all_in_one.py`는 `repro_bundle`을 `sys.path`에 먼저 등록해서
`new_bench_03` import를 이 폴더 기준으로 먼저 해석하도록 설정합니다.
