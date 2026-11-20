# GAM + past-only spatiotemporal kriging (TS=2,4,6)

새 폴더 `notebooks/gam_st_kriging`입니다. 기존 `kriging_gam`과 `window_spatial_gam`을 변경하지 않습니다.

## HPC 터미널

```bash
cd /mnt/pixstor/data/jlc3q/SatFire/TS-SatFire
git pull --ff-only origin main
cd notebooks/gam_st_kriging
conda activate satfire-audit
python -m pip install -r requirements.txt
python -m ipykernel install --user --name satfire-gam --display-name "TS-SatFire GAM"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TS_SATFIRE_RAW_ROOT=/home/jlc3q/data/SatFire/ts-satfire
```

HPC Jupyter의 할당된 계산 노드에서 `satfire_gam_st_kriging.ipynb`를 열고 TS-SatFire GAM 커널을 선택하세요. 첫 코드 셀에서 BUNDLE_DIR를 새 폴더의 절대경로로 지정할 수 있습니다. `DEMO_MODE=False`가 실제 자료 실행입니다. Python 설정은 Bash 터미널에 입력하지 마세요.

노트북 대신, 이미 할당받은 계산 노드에서 다음처럼 같은 파이프라인을 실행할 수도 있습니다.

```bash
MPLBACKEND=Agg python run_hpc.py --raw-root "$TS_SATFIRE_RAW_ROOT"
```

실제 HPC 속도/메모리는 측정되지 않았습니다. CPU4/RAM32GB는 초기 제안이며 스케줄러 요청 문법은 기관 환경에 맞춰야 합니다. GPU는 사용하지 않습니다. pykrige/pygam 설치는 필요하지 않습니다. 공분산 크리깅은 SciPy Cholesky 풀이로 구현했습니다.

## 무엇이 기존 실험과 다른가?

GAM 학습은 기존 알고리즘을 보존한 `gam_core.py`를 사용하고 공간항 없이 적합합니다. 새 모델은 잠재 Gaussian process에 과거 관측을 조건부로 넣어 다음 날 오차 효과를 실제 kriging 시스템으로 예측합니다. 단순 RBF 설명변수 추가나 입력 결측 복원은 아닙니다.

이항 타겟이므로 generalized/model-based kriging을 사용합니다:

- 고정 GAM logit offset + 블록/날짜별 latent GP.
- 각 픽셀의 다른 offset을 유지한 전체 Bernoulli likelihood.
- Laplace/Newton pseudo-response에 대한 covariance solve.
- 미래 latent mean/variance를 통해 Gauss–Hermite 적분한 확률.
- covariance는 공간거리(km)·시간차(day)에 대한 separable exponential kernel. 범위는 validation 후보 중 선택합니다. empirical variogram fitting은 아닙니다.

64×64 픽셀 블록에 공유 latent effect를 둡니다. 블록 중심 공분산은 계산 가능한 공간 근사이며 면적 적분한 정확한 block covariance가 아닙니다. GAM은 픽셀마다 달라 최종 확률 역시 픽셀마다 다릅니다. 고정 prior variance=1, 공간 범위 [5,20]km, 시간 범위 [1,3]일, 과거 이력 3일, 보정강도 [0,.5,1]은 사전 후보입니다. 시험 결과로 바꾸면 새 탐색 실험입니다. alpha=0이 validation 최적이면 GAM+STK는 GAM으로 환원되고 추가 이득 없음을 그대로 보고합니다. 모든 kriging 후보의 수렴 실패는 중단합니다.

## 누수와 비교 범위

38일 사건에서 target은 train 05-23–06-04, val 06-11–06-13, test 06-20–06-23입니다. 전 TS가 같은 날짜와 알려진 at-risk 픽셀을 사용하고 pixel-ID SHA256을 확인합니다.

GAM preprocessing/fit은 train만 사용합니다. 크리깅에는 **GAM 학습 종료일 이후의 예측에 대한 과거 관측**만 쓰므로 in-sample residual을 만들지 않습니다. 첫 validation 예측은 06-08–06-10의 관측으로 조건부 추정하고, 첫 test 예측은 06-17–06-19의 관측으로 조건부 추정합니다. 다음 날에는 새로 도착한 전날 관측을 추가합니다. 이전 test일 라벨을 다음 날 conditioning에 사용하는 **daily rolling-origin** 프로토콜입니다. GAM/covariance hyperparameters/threshold는 test에서 다시 선택하지 않습니다.

공통 TS 입력+target 블록은 서로 겹치지 않지만 누적 화재 상태·최근 관측 이력은 과거에서 이어집니다. 고정 train 데이터만 쓰는 다일 일괄 예측 또는 독립 사건 holdout으로 해석하면 안 됩니다. GAP 날짜도 이전에 관측되었으면 conditioning에 사용하므로 GAM과 결합모형은 서로 같은 고정 fitting label만 사용하는 구조는 아닙니다.

라벨은 기존 legacy_nan_zero를 유지한 신규 **기록** 화재입니다. 관측 부재를 실제 무화재로 확인한 것이 아닙니다. 원천 합성제품의 publication time과 upstream 미래정보 사용 여부는 코드 테스트로 확인할 수 없어 운영 예측 누수까지 제거됐다고 주장하지 않습니다.

## 결과와 검증

`runs/real_<UTC>/` 아래 새 결과를 저장합니다. `selection_before_test.json`과 seal로 test 전 선택을 고정합니다. 날짜별 conditioning audit에 target와 사용한 관측 날짜·수·Laplace 수렴 오차를 남깁니다. `predictions/*.npz`에는 공통 ids/y, GAM과 GAM_STK 확률, latent mean/sd를 저장합니다. 잠재 표준편차는 GP조건부 Laplace 근사만 반영하며 검증된 예측구간이 아닙니다.

`gam_core.py`는 이전 버전의 독립 복사본이며 새 모듈은 그 GAM/자료처리 함수만 사용합니다. 소스 해시와 버전은 run metadata에 기록합니다. 실행별 `base_predictions`/`observations` 캐시는 서로 다른 모델 또는 실행 간에 섞지 마세요.

```bash
python test_st_kriging.py
# 작은 합성 자료로 실제 GeoTIFF I/O와 전체 파이프라인을 확인:
MPLBACKEND=Agg python run_hpc.py --demo
```

합성 실행은 실제 사건의 성능 증거가 아닙니다. 검증 범위는 VALIDATION.md를 참고하세요.
