# 이틀 입력 NDVI·확산 진단 노트북

파일: `notebooks/ts_satfire_two_day_ndvi.ipynb`.
목적은 (1) 전체 FP 영상의 실제 크기/정합 확인, (2) 입력 t-1,t와 다음 날 t+1의
NDVI 비교, (3) 관측상 확산 없음과 라벨 불명 구분입니다. 모델 학습은 하지 않습니다.

## HPC에서 열기

```bash
conda activate satfire-audit
git pull --ff-only origin main
conda install -c conda-forge jupyterlab ipykernel pandas matplotlib nbformat nbclient -y
python -m ipykernel install --user --name satfire-audit --display-name "Python (satfire-audit)"
```

HPC가 제공하는 Jupyter/Open OnDemand 등에서 저장소의 노트북을 열고
Python (satfire-audit) 커널을 선택합니다. GPU는 필요하지 않습니다.
노트북을 저장소 루트 또는 notebooks 폴더를 작업 디렉터리로 하여 실행합니다.

기본 DATA_ROOT는 `/home/jlc3q/data/SatFire/ts-satfire`이며 EVENT_ID/ANCHOR_DATE를
설정 셀에서 선택할 수 있습니다. None이면 이틀 이력과 다음 날 관측이 가능한 첫 사례를
선택합니다. ROI 시작일/첫 관측일을 실제 발화일로 단정하지 않습니다.

## 먼저 확인할 설정

- NDVI_SOURCE='provided': 기본 FirePred 1번 밴드. 전체 27채널 표의 9번과 구분합니다.
  실제 밴드 의미·단위·스케일은 원시값 표와 로더로 확인합니다.
- NDVI_SCALE=0.0001, NDVI_OFFSET=0.0: VNP13A1 원본 10,000배 정수 단위를 복원합니다.
  근거는 [VNP13 사용자 가이드 Table 12](https://lpdaac.usgs.gov/documents/1372/VNP13_User_Guide_ATBD_V2.1.2.pdf)와
  현재 FP 로더의 NDVI 채널 평균 약 5110.55입니다. 이는 자동 min-max 정규화가 아닙니다.
  PROVIDED_NDVI_RAW_VALID_RANGE=(-10000,10000) 밖의 fill 등은 결측으로 제외합니다.
  이미 실수 NDVI로 변환한 export는 scale=1.0과 raw range=(-1,1)을 사용하십시오.
  이중 스케일링이 의심되면 `ndvi_may_already_be_scaled`를 표시하며 자동 재추정하지 않습니다.
  None으로 지정하면 파일 메타데이터를 사용하지만, 메타데이터가 누락된 원시 export에는 부적합합니다.
- NDVI_SOURCE='computed': 주간 I1/I2가 확인된 반사도이고 스케일이 적용됐는지 검증한
  뒤 REFLECTANCE_CONFIRMED=True를 설정합니다. 복사휘도에 반사도 지수를 무조건 적용하지 않습니다.
- BA_RULE=None: 라벨 인코딩을 확인할 때까지 unknown. 유한 코드=소실/NaN=비소실이면
  'finite', 유효 0/양수 인코딩이면 'positive'를 선택합니다. BA_BAND=8도 실제 파일로 확인합니다.
- DEMO_MODE=True는 합성 검증 사례를 별도로 생성합니다. 실제 HPC 분석 결과가 아닙니다.

## 영상 크기와 정합은 다릅니다

노트북은 기본적으로 FP로 선택된 모든 GeoTIFF의 헤더를 읽습니다.
height/width별 파일 수와 modality별 CRS/transform 불일치 사유를 별도 표시합니다.
동일 shape라도 원점·픽셀 크기·방향 또는 CRS가 다르면 같은 위치로 확인되지 않습니다.
SCAN_ALL_HEADERS=False로 설정하면 선택 사건만 확인하며 전체 검사로 보고하지 않습니다.
CRS를 강제로 붙이거나 원본 영상을 수정/재투영하지 않습니다.

CLI도 격자 실패를 `grid_differences.csv`에 파일 쌍·width/height·CRS·transform·원인별로
기록하도록 보완했습니다. 새 실행에서 `summary.json`의 `grid_failure_reasons`와
`issues.json`의 격자 실패 요약도 확인할 수 있습니다.

```bash
python scripts/audit_ts_satfire.py --task pred \
  --out audit_results/fp_grid_detail_001 --lengths 2 --check-grids
```

## 확산이 매일 없을 때

- 유효한 연속 소실 라벨에서 신규 소실이 0이면 observed_no_increment로 남깁니다.
- 두 날짜의 라벨이 없거나 인코딩/격자가 불명이면 unknown으로 남기고 0을 부여하지 않습니다.
- 일부 픽셀만 비교 가능하면 partial_coverage로 표시하고 그 픽셀만 향후 loss에 사용합니다.
- 전체 NaN이 비소실인지 미주석인지 구분할 근거가 없으면 보수적으로 unknown입니다.
- 모든 음성 날짜를 버리면 확률 추정이 편향될 수 있습니다. 학습 시 샘플링을 조정하더라도
  검증·시험은 정의한 대상 모집단을 유지하고 사건 단위 분할을 지킵니다.
- t+1 NDVI는 사후 비교용이며 예측 입력에 포함하면 안 됩니다.
- NDVI 감소 자체를 산불 확산으로 정의하지 않습니다. 제공 NDVI 제품이 시간 합성이라면
  일별 파일에서 값이 반복될 수도 있습니다.

실제 산불에 대한 무확산/확산 판정은 제공 BA의 인코딩을 확인한 뒤 해석해야 합니다.
legacy 누적 AF/합집합·crop과 동일한 벤치마크를 재현했다는 의미가 아닙니다.

## 결과와 검증

`audit_results/ndvi_two_day/`에 전체 헤더, 동일 날짜 격자 비교, 선택 밴드 통계,
사건별 일별 신규 소실 상태, 그림, provenance JSON을 저장합니다.
보고서는 재실행 시 갱신하며 원본은 변경하지 않습니다.

작성 환경에서는 합성 데이터로 노트북 전체 셀을 실행하고 그림을 확인했습니다.
실제 HPC 데이터에는 접근하지 못했으므로 배포 노트북에는 실제 결과나 합성 결과를
끼워 넣지 않았습니다. 실제 shape/NDVI/확산 빈도는 HPC 실행 후 확인합니다.

자동 실행이 필요한 경우 활성화한 HPC 환경에서:

```bash
python -m jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.timeout=1800 \
  --output ts_satfire_two_day_ndvi.executed.ipynb \
  notebooks/ts_satfire_two_day_ndvi.ipynb
```

설정 셀을 먼저 확인하십시오. 커널은 환경의 python3 커널로 실행됩니다.
문제가 있으면 Jupyter UI에서 등록한 satfire-audit 커널을 선택해 실행합니다.

테스트:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## 시간 해상도 주의

VNP13A1은 16일 합성 자료를 8일 간격으로 생산하는 제품입니다.
일별 FirePred 파일의 NDVI가 같을 수 있으며, 값이 같다는 사실이 산불 미확산을 뜻하지 않습니다.
합성 기간과 실제 배포 시점을 확인하기 전에는 제공 NDVI를 당시 이용 가능한 일별 독립 관측으로
간주하지 마십시오. NDVI_SCALE 수정과 BA_RULE 확정은 서로 다른 작업입니다.

## Git 업데이트 후 이전 함수가 남는 문제

Jupyter 커널은 파일이 바뀌어도 `from satfire_explore import read_ndvi`로 가져온 함수를
자동 교체하지 않습니다. 최신 노트북의 초기 import 셀은 모듈을 reload하고 함수 인자를
출력합니다. plot_pair도 실행마다 모듈을 다시 로드하고 그 모듈의 read_ndvi를 직접 사용합니다.
따라서 이전 전역 read_ndvi 별칭이 남아 있어도 사용하지 않습니다.

파일 경로가 다른 저장소를 가리키거나 디스크의 함수가 raw_valid_range를 지원하지 않으면
명확한 RuntimeError로 업데이트 필요성을 안내합니다. 인자를 지워서 오류를 숨기지 않습니다.
이 변경을 처음 적용할 때는 최신 노트북을 다시 열고 커널 재시작 후 위에서부터 실행하십시오.
