# TS-SatFire: 학습 전 데이터 진단

이 도구는 원본 GeoTIFF를 수정하지 않습니다. GPU는 필요하지 않습니다.
기본 데이터 경로: `/home/jlc3q/data/SatFire/ts-satfire`.
기존 agent/legacy 학습 코드를 실행하거나 변경하지 않는 독립 CLI입니다.

## HPC 실행

저장소 루트에서 실행합니다. Python 3.9 이상을 권장합니다.

```bash
# rasterio가 현재 환경에 없다면 설치합니다.
python -m pip install -r requirements-audit.txt
python scripts/audit_ts_satfire.py \
  --root /home/jlc3q/data/SatFire/ts-satfire \
  --out audit_results/run_001
```

출력 폴더는 원본 데이터 폴더 밖의 새 폴더 또는 빈 폴더여야 합니다.
기존 보고서가 있는 폴더는 덮어쓰지 않습니다.
HPC 정책에 따라 CPU 계산 노드에서 실행하십시오.

rasterio 없이 파일 목록부터 확인하려면:

```bash
python scripts/audit_ts_satfire.py \
  --out audit_results/inventory_001 --inventory-only
```

기본 입력 길이는 1, 2, 4, 6, 10일입니다. 3일도 함께 비교하려면:

```bash
python scripts/audit_ts_satfire.py \
  --out audit_results/run_002 --lengths 1 2 3 4 6 10
```

## 결과

| 파일 | 내용 |
| --- | --- |
| inventory.csv | 전체 GeoTIFF 파일, 사건, modality, 날짜, 바이트 |
| events.csv | VIIRS_Day 기준 관측 범위, 관측일 수, 내부 누락일 수 |
| windows.csv | 입력 길이별 파일 존재 조건을 만족하는 예측 후보 수 |
| metadata.json | 표본 GeoTIFF의 밴드 수/설명, CRS, 격자, NoData, scale/offset, 태그 |
| issues.json | 날짜 또는 modality 미해석, 중복 날짜, 메타데이터 읽기 오류 |
| summary.json | 전체 파일 수/용량, 평균·중앙 관측 범위, 이슈 수 |

## 해석 시 주의

- `mean_observation_span_days`는 관측 파일의 첫날~마지막 날(양 끝 포함) 평균입니다.
  실제 산불 지속 기간이나 라벨이 유효한 연속 일수와 다릅니다.
- `missing_days`는 첫 관측일과 마지막 관측일 사이의 누락만 셉니다.
  사건 시작 전/종료 후의 누락은 사건 메타데이터 대조 없이는 알 수 없습니다.
- 날짜는 파일 이름의 YYYY-MM-DD, YYYY_MM_DD 또는 YYYYMMDD에서 읽습니다.
  서로 다른 날짜가 둘 이상 있거나 해석 불가능한 파일은 추측하지 않고 issues에 기록합니다.
- VIIRS_Day/VIIRS_Night/FirePred 폴더 앞 경로를 사건 식별자로 사용합니다.
  중첩 export 구조나 예상과 다른 폴더 이름은 첫 결과를 보고 확인해야 합니다.
- 길이 L인 후보는 목표 날짜에 VIIRS_Day가 있고, 직전 L일 모두에
  VIIRS_Day와 FirePred 파일이 있는 경우입니다. 야간 파일은 별도 필수 조건으로 두지 않습니다.
- 중복 날짜는 날짜 집합으로 한 번만 세며, 해당 사건을 별도 표시합니다.
  중복 파일 중 어떤 것을 쓸지는 결정하지 않습니다.
- 후보 수는 **라벨 검증 전**의 수입니다. 영상 손상, 격자 불일치, 결측 픽셀,
  소실 라벨의 저장 규칙, 연속 날짜의 유효 라벨, 양성 픽셀 수는 추가 확인이 필요합니다.
- 밴드 의미를 임의로 확정하거나 NaN을 비화재 0으로 바꾸지 않습니다.
- 메타데이터는 기본 modality별 최대 30개를 경로 순서에서 분산 표본 추출합니다.
  `--metadata-limit 0`이면 모든 파일의 헤더를 읽습니다. 픽셀 통계는 계산하지 않습니다.
- NoData의 NaN/Inf는 JSON에서 문자열로 보존하고, 지정 없음은 null로 기록합니다.
- 지수/정답 생성과 모델 학습은 포함하지 않습니다. 산불 사건 단위 데이터 분할을
  확정한 뒤 진행하십시오.
- 보고서에는 로컬 경로가 포함됩니다. 필요 없는 원본 데이터나 실행 결과를 git에 추가하지 마십시오.

## 테스트

```bash
python -m unittest discover -s tests -p 'test_ts_satfire_audit.py' -v
```

날짜 파싱, 누락/중복과 윈도 계산, 재실행 덮어쓰기 방지,
합성 GeoTIFF의 격자/밴드/NaN NoData 보존을 확인합니다.
rasterio가 없으면 GeoTIFF 테스트는 skip됩니다.

## 첫 실행 후 확인 순서

1. summary.json과 issues.json으로 파일 구조/날짜 인식 여부 확인.
2. events.csv와 windows.csv로 입력 기간별 데이터 가용성 비교.
3. metadata.json을 실제 로더와 대조하여 밴드와 격자 확인.
4. 대표 사건의 연속 라벨을 시각 확인한 뒤 신규 소실 정답 생성 단계 구현.
