# 학습된 PLY로 시작하는 ProDiG + GeoQuery

English: [USAGE.md](USAGE.md).

기존 **baseline(`gsplat_geoquery_sklearn`) 실행과 동일한 파이프라인**입니다 — GeoQuery가
**모든 고도 단계에서 원본 GT 항공 사진과 그 depth를 reference로** 사용해 novel view를
보정합니다. 한 가지가 추가되었습니다: COLMAP point cloud 대신 **이미 학습된 Gaussian PLY**
로 시작할 수 있어, initial training을 건너뛰고 바로 렌더 → 보정 → 하강으로 들어갑니다.

```
입력 PLY (이미 품질이 좋은 모델)
      │
      ├─ 고도 90% 지점의 novel view 렌더링        → noisy_views/
      ├─ GeoQuery가 보정 (reference = GT 항공사진 + 그 depth)
      │                                          → fixed_views/
      ├─ 보정된 view로 학습
      └─ 70% → 50% → 30% → 10% 반복
```

## 필요한 입력

| 입력 | 필수 | 설명 |
|---|---|---|
| `--data <dir>` | 예 | COLMAP scene: `images/` + `sparse/0/`. 카메라 모델은 PINHOLE / SIMPLE_PINHOLE 이어야 합니다 (먼저 `colmap image_undistorter` 실행) |
| `--input-ply <file>` | 아니오 | 학습된 3DGS PLY. 생략하면 기존처럼 point cloud부터 학습합니다 |
| `--reference-depth-dir <dir>` | 예 | GT 이미지별 metric depth `.npy`. 파일명은 이미지 stem과 동일 (`H_010_00001.npy`) |
| `--aerofix-factory` | 예 | `examples.geoquery_bridge:create_geoquery` |
| GPU | 예 | CUDA. 1598×1064에서 약 13GB. 메모리는 Gaussian이 아니라 GeoQuery가 대부분 차지합니다 |

**reference depth가 반드시 필요한 이유.** GeoQuery는 geometry-conditioned 모델이라,
reference view를 그 view의 depth를 통해 target 시점으로 warp합니다. 즉 depth는
**보정 대상 view가 아니라 reference 이미지의 depth**가 필요합니다. 없으면 모든 view가
soft-reject 되어 결과물이 하나도 생성되지 않습니다.

GeoQuery 체크포인트 경로 (필수, 기본값 없음):

```bash
export GEOQUERY_CHECKPOINT=/path/to/geoquery_window_size3_down1.pkl
export GEOQUERY_SOURCE_DIR=/path/to/GeoQuery/src
```

### `anchor_prodig` — GeoQuery / AeroFix bridge에만 필요

`anchor_prodig` 는 연구실 연구 저장소의 메인 패키지이며, 프로젝트가 개발 중인 동안
비공개입니다. PyPI에 없습니다. 이 파이프라인은 여기서 딱 한 가지만 가져다 씁니다 —
diffusion refiner 체크포인트를 로드하고 실행하는 코드입니다:

```python
from anchor_prodig.config import DiffusionConfig          # refiner 설정
from anchor_prodig.diffusion.aerofix import create_refiner # GeoQuery / aeroFix 생성
from anchor_prodig.diffusion.base import RefinementRequest # refinement 요청 1건
```

그 외의 모든 것 — Gaussian 학습, rasterization, distance-adaptive gating,
progressive 하강 — 은 이 패키지 안에 자체적으로 들어 있고 의존하지 않습니다.
`examples/geoquery_bridge.py` 와 `examples/real_aerofix_bridge.py` 만 import 하며,
기본값인 `--aerofix-factory identity` 와 `examples/aerofix_bridge_example.py` 는
없이도 실행됩니다.

```
prodig/                    파이프라인 (자체 완결)
        │ 호출
examples/*_bridge.py       얇은 어댑터
        │ import
anchor_prodig.diffusion    refiner 체크포인트 로드 + 실행 (연구실 패키지)
```

연구실 저장소에 접근할 수 있다면, 그 checkout에서 설치하고 확인하세요:

```bash
pip install -e /path/to/lab-repo   # src/ 의 anchor-prodig 패키지 설치
python -c "import anchor_prodig; print(anchor_prodig.__file__)"
```

### ⚠ scikit-learn 은 필수 — 새로 설치할 때마다 먼저 확인

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

실패하면 **무엇을 실행하기 전에 먼저 설치**하세요:

```bash
pip install scikit-learn
```

`requirements.txt` 와 `pyproject.toml` 에 선언되어 있어서 `pip install -e .` 로는
설치되지만, `PYTHONPATH=.` 로만 실행하면 설치되지 않습니다. `--no-deps` 로 editable
설치한 경우에도 빠집니다. (`scikit-image` 는 다른 패키지이며 대체되지 **않습니다**.)

**왜 이렇게 엄격하게 확인하는가.** 원래 `prodig/model.py` 는
`sklearn.neighbors.NearestNeighbors` 를 아무 메시지도 출력하지 않는
`try/except Exception:` 안에서 import 했습니다. 없으면 point별 k-NN 추정 대신 모든
Gaussian이 **동일한** 초기 scale을 갖게 됩니다. park scene 실측:

| | 초기 scale 중앙값 | gsplat prune 임계값 초과 |
|---|---|---|
| scikit-learn 있음 | 0.00103 | 336 / 576,408 = **0.06 %** |
| 없음 | 0.2303 (전부 동일) | **100 %** |

gsplat은 oversized pruning을 `step > reset_every` 뒤에서만 수행하므로, 3,000 step
이후 첫 refine 단계에서 모델 전체가 삭제됩니다 — 실측 **569,729 → 11개**.
렌더는 검게 나오고, refiner는 검은 입력으로부터 어두운 출력을 만들며, 그 어두운
이미지가 다시 학습 target이 됩니다. 그리고 이 run은 **exit code 0으로 정상 종료**되며
어떤 에러도 발생하지 않습니다. 유일한 증상은 결과 이미지가 전부 어둡다는 것뿐입니다.
지금은 scikit-learn 이 없으면 시작 시점에 `ImportError` 가 발생합니다.

## 실행 방법

`examples.` 를 찾을 수 있도록 **저장소 루트에서** 실행하세요.

### 학습된 PLY로 시작 (initial training 없음)

```bash
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m prodig.train \
  --data data/park \
  --output outputs/<실행이름> \
  --input-ply /path/to/model.ply \
  --input-ply-frame world \
  --aerofix-factory examples.geoquery_bridge:create_geoquery \
  --reference-depth-dir data/park/depths_metric \
  --min-aerofix-quality 0.01 \
  --stage-steps 7000 \
  --factors 0.9,0.7,0.5,0.3,0.1 --trajectory forward \
  --max-views-per-stage 0 \
  --refine-stop-iter 12000 --refine-every 300 \
  --grow-grad2d 0.0008 --prune-opa 0.02 --checkpoint-every 1000
```

`--input-ply` 를 주면 `--initial-steps` 는 자동으로 **0** 이 됩니다. 항공 뷰로 조금 더
맞추고 싶으면 값을 직접 지정하세요.

### PLY 없이 (baseline 실행과 완전히 동일)

`--input-ply*` 를 빼고 `--initial-steps 7000` 을 넣으면 됩니다.

## `--input-ply-frame` — 반드시 맞게 지정

이 파이프라인은 scene을 단위 구(unit sphere)로 정규화하므로, PLY는 둘 중 하나의 좌표계입니다.

| PLY 출처 | 사용할 값 |
|---|---|
| 외부 학습기 (graphdeco 3DGS, Scaffold-GS 등), COLMAP 좌표계 | `world` (기본값) |
| 이 파이프라인이 만든 `plys/*.ply` | `normalized` |

잘못 지정해도 **에러가 나지 않습니다** — Gaussian이 카메라에서 멀리 떨어진 곳에 놓여
렌더 결과가 전부 비어버립니다. 실행 시작 시 아래와 같은 확인 문구가 출력되니
지정한 값과 일치하는지 꼭 확인하세요.

```
PLY: 6,043,753 Gaussians, extent radius 1.94; normalized camera radius 1.00 -> looks like frame='normalized'
```

## 자주 빠뜨리는 필수 옵션

- **`--min-aerofix-quality 0.01`** — view를 사용하기 위한 refiner quality 최솟값.
  quality 0 (refiner의 명시적 거부)은 항상 제외됩니다. 이 처리가 들어가기 전에는
  `0.0 < 0.0` 이 거짓이라 기본값 `0.0` 에서 거부된 view가 걸러지지 않았습니다.
- **`--reference-depth-dir`** — 위 설명 참고. 없으면 결과물이 생성되지 않습니다.

## 출력물

```
noisy_views/stage_XX_YY/   GeoQuery 보정 전 원본 렌더
fixed_views/stage_XX_YY/   GeoQuery 보정 후
plys/                      initial + 단계별 + final (raw, 해당되면 _gated)
checkpoints/               initial.pt, 단계별, final.pt, latest.pt
logs/train.jsonl           50 step마다 loss와 Gaussian 개수
config.json, run.log
```

park scene에서는 depth가 975장 중 855장에만 있습니다 — 나머지 120장은 calibrated depth가
없어 매 단계에서 제외됩니다. 버그가 아니라 정상 동작입니다.

## 참고 사항

- **해상도는 데이터셋을 따릅니다.** resize 옵션이 없어 COLMAP 카메라 크기 그대로
  렌더링합니다. `data/park` → 1598×1064, `data/park_lowres` → 576×384.
- **GeoQuery는 긴 변 기준 1024px로 제한**됩니다 (`GEOQUERY_MAX_SIDE`). 제한 없이
  1598×1064로 돌리면 24GB 카드에서 약 23.5GB가 필요합니다. 렌더와 `noisy_views` 는
  원본 해상도를 유지하고, 보정만 작게 계산한 뒤 다시 확대합니다.
- **이어서 실행**은 `--resume <checkpoint>`. `latest.pt` 가 1000 step마다 저장됩니다.

## 테스트

```bash
PYTHONPATH=. python -m pytest tests/ -q     # 38개
```
