# HPC 실행 안내

## 파일

- `satfire_kriging_gam.ipynb`: 실제 HPC 자료가 기본값인 14-cell 실험 노트북.
- `satfire_pipeline.py`: 검증 가능한 수치 처리 함수.
- `satfire_demo.py`: 별도 합성 GeoTIFF 생성; 실제 사건으로 대체하지 않음.
- `requirements-satfire.txt`: 설치 목록. 실행 시 실제 버전 provenance 저장.
- `SOURCE_AUDIT.md`: 확인한 원본 코드/자료 계약과 확인하지 못한 사항.
- `VALIDATION.md`: 로컬 검증 기록.

## 사용 순서

1. 이 폴더의 파일을 같은 HPC 디렉터리로 복사합니다. 원본 TS-Agentic-AI 코드는 수정할 필요 없습니다.
2. 노트북 0절의 별도 conda 환경을 만들고 HPC compute allocation에서 해당 kernel을 선택합니다. CPU 실행이며 실제 자료 실행 시간/최대 메모리는 아직 측정하지 않았습니다. 처음에는 4 CPU/16 GB 정도를 파일럿 자원 요청 예시로 고려하되 사이트 정책에 맞게 조정합니다.
3. `BUNDLE_DIR`와 `CFG['raw_root']` 확인. 기본 raw root는 `/home/jlc3q/data/SatFire/ts-satfire`이며 그 아래 사건 폴더를 찾습니다. `DEMO_MODE=False` 유지.
4. 1절까지 실행합니다. `runs/real_.../inventory.csv`, `audit.json`에서 날짜, 격자, band descriptions, scale/offset, nodata, missing fractions를 확인합니다.
5. 타겟 NaN 의미와 입력자료 이용 가능 시각을 확인하여 config의 두 REVIEWED flag, TIMING_NOTE, label_policy_reason을 작성합니다. 필요할 때만 `value_rules`를 설정합니다. 아직 생산 규칙을 모르면 flag를 True로 만들지 않습니다.
6. 아래 셀부터 이어서 실행합니다. 설정 셀을 다시 실행하면 새로운 run 폴더를 만듭니다. 외부 변경한 helper는 kernel restart 후 읽습니다.

첫 로컬 검증은 `DEMO_MODE=True`로 가능하지만 synthetic 결과는 실제 산불 성능이 아닙니다. 실데이터 경로가 없으면 명시적으로 실패합니다.

## 해석

이 사건 자체의 앞부분을 학습에 쓰는 exploratory pilot입니다. 원래 FP test event를 개발에 사용했음을 기록하고 공식 독립 시험에서 제외해야 합니다. 이 설정으로는 California 일반화나 agent 협업 성능을 주장할 수 없습니다.

확률은 AF/BA 관측 누적 영역의 다음 날 신규 기록 확률입니다. `unknown`은 0%가 아니고 타겟은 kriging으로 복원하지 않습니다. known label 영역에서만 평가 지도가 출력됩니다. cloud QA가 없는 파일에서는 nodata 복원만 가능하며 kriging으로 구름을 탐지하지 않습니다.

## 주요 결과

- `inventory.csv`, `audit.json`, `review_decisions.json`, `provenance.json`
- 지표별 `*_restoration.tif`: 값 / 조건부 kriging 분산 / 입력 결측 / 복원 성공
- `restoration_validation.csv`: 숨긴 블록의 오차와 동일 위치 IDW 비교
- `split_manifest.csv`, `feature_names.json`
- `metrics.csv`, `calibration_bins.csv`, `test_predictions.csv.gz`
- `*_prediction.tif`, `*_panels.png`, `calibration.png`
- `*_selection.json`, `*.pkl` (본인이 생성한 신뢰 가능한 pickle만 열 것)
- `completion.json`

## 지표 추가

`CFG['features']`에 SPECS의 연속형 또는 circular 지표를 추가할 수 있습니다. 모든 지표를 무조건 사용하는 것은 아니며 원자료 단위·가용 시각을 확인합니다. 방향은 sin/cos로 분해; 범주형 landcover는 잘못된 수치 spline/kriging을 피하려고 현재 차단합니다. landcover를 추가하려면 one-hot/factor encoding과 별도 결측 정책을 구현해야 합니다. 파생 NDMI/NBR 등은 해당 센서 밴드·단위·정의를 검증한 뒤 추가해야 하며 이번 버전에 임의 계산하지 않았습니다.
