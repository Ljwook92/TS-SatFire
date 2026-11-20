# TS=2/4/6 logistic GAM vs spatial logistic GAM

새 폴더에 압축을 풀고 실행하세요. 기존 kriging_gam 노트북/helper를 덮어쓰지 않습니다.

## 터미널에서 실행

```bash
conda activate satfire-audit
python -m pip install -r requirements.txt
python -m ipykernel install --user --name satfire-gam --display-name "TS-SatFire GAM"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TS_SATFIRE_RAW_ROOT=/home/jlc3q/data/SatFire/ts-satfire
```

HPC의 기존 Jupyter 실행 방식으로 `satfire_window_spatial_gam.ipynb`를 열고 **TS-SatFire GAM** 커널을 선택하세요. Python 설정 `DEMO_MODE = False`와 `CFG[...]`는 터미널이 아니라 노트북 셀에서 실행합니다. BUNDLE_DIR는 window_gam.py가 있는 폴더입니다.

이 번들은 pykrige/pygam에 의존하지 않습니다. 모델은 sklearn의 cubic B-spline basis + weighted penalized logistic likelihood로 구현했습니다. 공간 모델은 동일 설계행렬에 Gaussian RBF 공간항을 추가하여 함께 적합합니다. ridge coefficient penalty이며 2차 차분 spline penalty, 잔차 크리깅 또는 완전한 GP posterior와 동일하지 않습니다.

## 실행 결과

새 `runs/real_<UTC timestamp>/`를 매번 만들고 기존 결과에 덮어쓰지 않습니다. 입력 데이터는 읽기 전용입니다. 실제 사건 기본 캐시는 수백 MB 이상이며 모델/예측 추가 공간이 필요합니다. CPU4/RAM32GB는 시작 제안일 뿐 실제 HPC 사용량은 아직 측정하지 않았습니다.

- inventory.csv, source_files.csv: band 품질/경로/크기/mtime (원천 파일 내용 해시는 아님).
- config.json, versions.json, grid.json: 설정·실행환경·좌표.
- evaluation_manifest.csv: 최대 TS 기준 공통 target 날짜, 픽셀 수·SHA256.
- training_rows.npz: 전체 양성 + 추출 음성, 역확률 가중치; 모든 모델 동일.
- tuning.csv, selection_before_test.json, *.joblib: validation 선택과 모델. joblib은 신뢰하는 본인 출력만 열 것.
- test_metrics.csv, test_metrics_by_date.csv, calibration.csv: 동일 시험 픽셀 평가.
- paired_model_differences.csv: 각 TS에서 공간항 추가의 지표 변화. 통계적 유의성 검정이 아님.
- predictions/*.npz, comparison.png, test_maps.png: 픽셀별 예측과 그림.

검증 또는 시험에서 라벨 양성이 없는 날짜도 제거하지 않습니다. 날짜별 AP는 양성이 없으면 결측으로 기록합니다. 전체 validation에 양성과 음성이 모두 없으면 임계값 선택이 불가능하여 중단합니다. 테스트 클래스가 하나면 ROC-AUC는 결측입니다.

## 재현/한계

기본 38일 자료: train target 05-23–06-04, val 06-11–06-13, test 06-20–06-23. 가장 긴 TS 입력을 각 블록 안에 제한하므로 TS2도 같은 target만 사용합니다. 이전 노트북의 test 날짜/지표와 직접 대조하지 마세요. 원 논문의 TS 실험은 신경망에 시계열을 넣는 실험이지만 여기서는 last/mean/change/missing-fraction을 사용합니다.

동일 사건의 과거 좌표 효과가 이후에 도움이 되는지를 평가합니다. 누적 화재는 전날까지 업데이트하므로 one-step daily evaluation입니다. 시험 첫날의 상태만으로 이후 모든 날을 예측하는 실험이 아닙니다. 파일 날짜가 제품의 실제 공개 시각을 보장하지 않으므로 운영 예측 주장을 하지 않습니다.

기존 감사의 legacy_nan_zero 정책을 유지합니다: AF/BA 유효 양성의 누적 기록, NaN은 양성 기록 없음. 관측 부재가 실제 비화재를 뜻하지 않습니다. 순수 BA 확장이나 최초 발화 타겟과 다릅니다. 설명변수 결측은 train 중앙값으로 채우며, test 결측 여부에 따라 평가 픽셀을 제외하지 않습니다.

## 합성 검증

```bash
python test_window_gam.py
```

실제 데이터는 이 작성 환경에서 접근할 수 없습니다. 합성 검증은 날짜 분리, 미래자료 차단, 픽셀 일치와 파이프라인 작동을 검사하며 실제 예측 성능의 증거가 아닙니다. 새 커널 전체 실행 결과는 `VALIDATION.md`에 기록합니다.
