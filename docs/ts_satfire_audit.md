# TS-SatFire: 확산 예측(FP) 데이터 진단

원본 GeoTIFF를 수정하지 않는 CPU 진단 도구입니다. GPU는 필요하지 않습니다.
기존 `--task all` 전체 조사 기능은 유지하며, FP 분석에는 `--task pred`를 사용합니다.

## Miniconda / HPC

저장소 루트에서 실행합니다. 환경 생성은 최초 한 번만 필요합니다.

```bash
conda create -n satfire-audit -c conda-forge python=3.11 rasterio -y
conda activate satfire-audit
git pull --ff-only origin main

python scripts/audit_ts_satfire.py \
  --task pred \
  --root /home/jlc3q/data/SatFire/ts-satfire \
  --out audit_results/fp_001 \
  --lengths 1 2 3 4 6 10 \
  --check-grids
```

기본 입력 길이는 1, 2, 4, 6, 10일입니다. 논문과 같은 2, 4, 6일만 보려면
`--lengths 2 4 6`을 사용합니다. HPC 정책에 따라 CPU 계산 노드에서 실행하십시오.
출력은 원본 데이터 밖의 새 폴더/빈 폴더로 지정해야 합니다.

rasterio가 없는 상태에서 파일 존재와 분할만 확인하려면:

```bash
python scripts/audit_ts_satfire.py --task pred \
  --out audit_results/fp_inventory_001 --inventory-only
```

## FP 사건 분할의 출처

- `legacy/roi/us_fire_2017_out_new.csv` ~ `us_fire_2020_out_new.csv`를 통합합니다.
- `legacy/scripts/dataset_gen_pred.py`의 첫 번째 문자열 목록 `val_ids`를 AST로 읽습니다.
  기존 스크립트를 import하거나 실행하지 않습니다.
- 2017~2020년 목록과 검증 ID의 교집합이 검증, 나머지가 학습입니다.
- `legacy/roi/us_fire_2021_out_new.csv`가 시험 사건 목록입니다.
- 현재 저장소 CSV 기준 학습 125, 검증 13, 시험 24건입니다. 검증 후보 15개 중
  `22713339`, `23301962`는 현재 CSV 목록에 없습니다. 실제 HPC 파일 존재 여부는 별도로 집계합니다.
- CSV에 없는 AF 시험 사건은 FP 집계에서 제외하고 `excluded_files.csv`에 남깁니다.
- 사건 폴더명은 CSV ID와 정확히 일치해야 합니다. 동일 ID의 서로 다른 상위 경로가
  둘 이상 있으면 임의 병합하지 않고 ambiguous로 표시합니다.
- `--roi-dir`와 `--split-script`로 다른 검증된 버전을 지정할 수 있습니다.
  사용한 입력 파일의 SHA256을 결과에 기록합니다.

## 출력

| 파일 | 내용 |
| --- | --- |
| summary.json | FP 파일 용량, 선언/실제 관측 사건 수, 분할별 관측 기간과 입력 길이별 후보 집계 |
| fp_splits.json | 분할 출처와 해시, 검증 후보 ID, CSV에 없는 검증 후보 |
| events.csv | 학습/검증/시험의 모든 선언 사건, 실제 관측 여부, 관측 범위, ROI 범위와의 차이 |
| windows.csv | 분할·사건·입력 길이·예측 날짜별 상태와 제외 사유 |
| inventory.csv | FP 사건으로 선택된 GeoTIFF 목록 |
| excluded_files.csv | FP 분석에서 제외된 파일과 사유 |
| metadata.json | FP 파일의 밴드·단위 태그·CRS·격자·NoData·scale/offset 표본 |
| issues.json | 날짜 해석 실패, 중복, 사건 경로 중복, 파일 읽기 오류 |
| label_band_stats.json | 선택적으로 검사한 소실 라벨의 유한/0/양수/음수 픽셀 통계 |

관측 기간은 **실제 산불 지속 기간이 아닙니다**. 원래 ROI 시작/종료일도 별도 기록합니다.
관측 파일이 ROI 범위를 벗어나더라도 날짜를 임의로 잘라내지는 않습니다.
첫/마지막 관측 사이의 누락과 ROI 범위 안의 누락을 별도 기록합니다.

## 윈도와 격자 검사

길이 L의 입력은 목표 날짜 직전 L개 **연속 달력 날짜**입니다. 누락 날짜를 건너뛰어
붙이지 않습니다. 목표일 VIIRS_Day와 과거 L일의 VIIRS_Day·FirePred가 필요합니다.
초기 관측 범위가 L일 미만인 목표는 집계 대상에서 제외합니다.

필요한 날짜에 중복 파일이 있으면 후보에서 제외합니다. 야간 영상은 기존 로더처럼
선택 입력으로 취급하되 누락 일수를 기록하며, 중복 야간 파일은 제외합니다.

`--check-grids`는 후보에 사용되는 파일들의 헤더를 읽어 CRS·transform·폭·높이를
비교합니다. 주간 최소 6밴드, 보조 최소 19밴드, 존재하는 야간 최소 2밴드도 확인합니다.
픽셀 값의 품질이나 모든 입력의 NoData 비율까지 검증하는 것은 아닙니다.
metadata는 기본 modality별 최대 30개 표본이며, `--metadata-limit 0`이면 모든 헤더를
기록합니다. 격자 검사에 필요한 헤더는 이 표본 제한과 무관하게 읽습니다.

## 소실 라벨 검사: 인코딩 확인 후 별도로 실행

실제 GeoTIFF의 밴드/NoData/인코딩을 확인하기 전에는 기본 밴드 번호를 가정하지 않습니다.
예를 들어 8번 밴드가 '유한 값=소실, NaN=비소실'이라는 것을 확인한 경우에만:

```bash
python scripts/audit_ts_satfire.py --task pred \
  --out audit_results/fp_ba_finite_001 \
  --lengths 2 4 6 \
  --check-labels --ba-band 8 --ba-rule finite
```

`--check-labels`는 격자 검사도 수행합니다. `positive` 규칙은 유한·유효 픽셀 중
양수를 소실, 0 이하를 비소실로 해석합니다. 실제 인코딩이 이 규칙에 맞는지 먼저 확인하십시오.
`finite` 규칙은 NaN을 비소실로 해석하지만, 밴드 전체가 비유한 값인 날짜는
라벨 미제공 가능성이 있으므로 `unannotated_label_pair`로 제외합니다.
finite 규칙과 유한 NoData 값(예: -9999, 0)이 충돌하면 추측하지 않고 오류로 기록합니다.

정답 픽셀 통계는 마지막 입력일과 목표일에서 비교 가능한 영역의
`BA(target) AND NOT BA(last_input)`입니다. 신규 소실·역으로 줄어든 소실 픽셀 수를 기록합니다.
positive 규칙의 부분 결측은 양쪽 날짜에서 유효한 픽셀만 비교합니다.

이 검사는 **지정된 BA 인코딩에 따른 연속 마스크 차이 진단**입니다. 기존 FP 로더의
누적 AF 또는 AF/BA 합집합, 사건별 label_sel, 256×256 crop과 동일한 벤치마크를
재현했다고 주장하지 않습니다. 해당 legacy label_sel은 events.csv에 보존합니다.
기본 라벨 검사는 전체 장면에서 계산하며 정답 파일을 생성하지 않습니다.

`passing_candidates`는 선택한 검사 단계를 통과한 수입니다. 픽셀 라벨 검사를 하지
않으면 `positive_label_windows`는 0이 아니라 null입니다. 어떤 단계도 수작업 검증된
정답이나 즉시 학습 가능한 샘플임을 보장하지 않습니다.

## 다음 확인 순서

1. summary.json과 fp_splits.json으로 사건 분할과 실제 데이터 가용량 확인.
2. issues.json과 windows.csv로 누락/중복/격자 문제 확인.
3. metadata.json, 원본 픽셀, legacy 로더를 대조해 라벨 인코딩 확정.
4. 대표 사건의 연속 마스크를 시각 검증하고, 사용할 정답 프로토콜 결정.
5. 사건 단위 분할을 유지하여 파생변수·모델 학습 진행.

보고서에는 HPC 경로가 들어갑니다. 원본/보고서를 불필요하게 git에 추가하지 마십시오.

## 테스트

```bash
python -m unittest discover -s tests -p 'test_*audit.py' -v
```

분할 CSV 교집합, legacy 코드 비실행, AF 사건 제외, 누락/중복 날짜,
누락 사건 보존, 격자 불일치, 명시적 BA 증분과 미주석 날짜 처리 및 기존 CLI를 검사합니다.
rasterio가 없으면 GeoTIFF 관련 테스트는 skip됩니다.

## 격자 원인 상세 및 NDVI 노트북

격자 실패는 `grid_differences.csv`에서 크기, CRS 누락/불일치, transform 불일치를
분리해 확인합니다. summary.json의 grid_failure_reasons는 고유 파일 쌍 기준이며
입력 윈도 실패 건수와 다릅니다. issues.json에도 격자 실패 요약이 포함됩니다.

전체 영상 크기 조사와 이틀 입력 NDVI 비교는 [노트북 안내](two_day_ndvi.md)를 참고하십시오.
