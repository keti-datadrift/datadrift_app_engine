# 의류 세그멘테이션 벤치마크

작성일: 2026-09-18

추가 학습 없이 공개 가중치 세그멘테이션 모델이 상의 상품 이미지에서 판매 대상 의류를 얼마나 잘 찾고, 여러 의류 중 실제 제품을 고르는지를 비교한다. 입력 WebP는 읽기 전용이며 산출물은 `artifacts/garment_segmentation/<run_id>`에만 기록한다.

1차는 픽셀 정답 없이 진행한다. 보고 가능한 것은 타깃 선택 성공률, 블라인드 육안 품질, 지연시간/메모리, 라이선스 적격성이다. mIoU, instance AP, Boundary F1은 보고하지 않는다.

`config.yaml`은 gitignore 대상이다. `config.example.yaml`을 복사한 뒤 로컬 `input_root`와 채널 폴더명(`disk_name`)을 채운다.

## 표본

- 카테고리: `tops`
- 450장 고정: dev 50 / primary 300 / stress 100
- primary는 source × subtype × shoot_type 층화
- stress는 모델컷·색 다양성·레이어드 키워드 등 복잡도 점수가 높은 이미지를 과표집
- 각 이미지에 타깃 존재 여부, 클릭 포인트, scene/occlusion/pose/view 주석 칸을 둔다

프로젝트의 `.venv`만 사용한다.

```bash
cp experiments/garment_segmentation/config.example.yaml experiments/garment_segmentation/config.yaml
# config.yaml의 input_root / sources.*.disk_name 을 로컬 값으로 수정
./.venv/bin/python experiments/garment_segmentation/run.py --freeze-only
```

## 모델 후보

1차는 SAM/DINO 없이 가벼운 세 계열만 비교한다.

| key | 계열 | 실제 체크포인트 | 기본 적격성 |
| --- | --- | --- | --- |
| `yolo` | 패션 검출 | YOLOS-Fashionpedia (`valentinafeve/yolos-fashionpedia`) | research-only |
| `segformer` | 의류 파싱 | SegFormer B2 clothes (`mattmdjaga/segformer_b2_clothes`) | research-only |
| `clipseg` | CLIP 텍스트 마스크 | CLIPSeg (`CIDAS/clipseg-rd64-refined`) | adoptable |
| `dummy` | 파이프라인 테스트용 | 없음 | 벤치마크 제외 |

`yolo`는 Ultralytics YOLOv8이 아니다. Fashionpedia 46-class ViT detector이고, 마스크는 박스 근사다. `segformer`는 픽셀 단위 의류 파싱이다. `clipseg`는 프롬프트 히트맵으로 마스크를 만든다.

SAM 계열(`grounded_sam21`, `sam3`, `owlv2_sam`)은 이 세 모델이 부족할 때만 추가한다.

코드 라이선스와 체크포인트 라이선스는 `licenses.py`에서 분리해 기록한다. 공개 가중치를 곧바로 오픈소스로 부르지 않는다. 기본값은 `download_weights: false`다. 첫 실행에서 가중치를 받으려면 `--download-weights`를 붙인다.

## CPU 1차 실행

표본은 이미 `tops-450-20260918`에 freeze되어 있다. 프로젝트 `.venv`만 사용한다.

1. 단위 테스트

```bash
./.venv/bin/python experiments/garment_segmentation/test_experiment.py
```

2. CPU 스모크 (dev 8장, 다운로드 허용, resolver 임베딩 생략)

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --reuse-dataset tops-450-20260918 \
  --models yolo segformer clipseg \
  --device cpu \
  --download-weights \
  --splits dev \
  --limit 8 \
  --skip-encoder \
  --run-id cpu-light-dev8
```

M2 Pro CPU 기준 첫 다운로드 포함 **수 분**, 이후 장당 대략 YOLO 2–4초, SegFormer 1–2초, CLIPSeg 1–3초로 보면 된다.

3. 스모크가 통과하면 dev 50장

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --reuse-dataset tops-450-20260918 \
  --models yolo segformer clipseg \
  --device cpu \
  --splits dev \
  --skip-encoder \
  --run-id cpu-light-dev50
```

가중치가 캐시되어 있으면 `--download-weights`는 생략해도 된다. 예상 **10–20분**.

4. 육안 확인 후 필요하면 primary/stress 전체

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --reuse-dataset tops-450-20260918 \
  --models yolo segformer clipseg \
  --device cpu \
  --run-id cpu-light-450
```

450장 × 3모델 CPU는 대략 **1–1.5시간**. FashionSigLIP resolver까지 쓰려면 `--skip-encoder`를 빼면 되고, 모델당 약 10–20분이 추가된다.

결과: `artifacts/garment_segmentation/<run_id>/report/index.html`, overlay는 `selected/<model>/combined/overlays/`.

타깃 Top-1을 계산하려면 `dataset/annotate.html` 또는 `annotation_template.csv`에 클릭 포인트를 채운 뒤 `--annotations`로 넣는다. 주석 전에는 생성률·지연시간·contact sheet만 본다.

이 세 모델의 타깃 선택과 육안 usable이 충분하면 SAM/DINO는 건너뛴다. 박스만 맞고 경계가 부족하거나, 레이어드/가림에서 실패가 크면 아래 SAM 전용 아티팩트를 사용한다.

## SAM 전용 아티팩트

경량 모델과 산출물을 섞지 않는다. 설정은 `config.sam.yaml`이다. 현재 로컬 설정은 `output_root: artifacts/garment_segmentation`이라 Grounded SAM 캐시(`sam-mps-450`)와 같은 트리에 쌓인다. 표본은 freeze `tops-450-20260918`을 재사용하고 원본 WebP는 그대로 둔다.

```bash
cp experiments/garment_segmentation/config.sam.example.yaml experiments/garment_segmentation/config.sam.yaml
```

포함 모델:

| key | 계열 | 적격성 |
| --- | --- | --- |
| `grounded_sam21` | Grounding DINO + SAM 2.1 | adoptable |
| `sam3` | SAM 3 open-vocabulary (`facebook/sam3`) | research-only |

`facebook/sam3`는 Hugging Face gated repo다. 웹에서 라이선스에 동의한 뒤 로컬에서 로그인한다. `facebook/sam3.1`은 transformers 연동이 없어서 이 벤치마크는 SAM 3를 쓴다.

```bash
./.venv/bin/hf auth login
# 토큰은 Read 권한이면 된다. 한 번만 하면 캐시된다.
```

3. SAM3 스모크 (dev 8장, 가중치 다운로드)

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --config experiments/garment_segmentation/config.sam.yaml \
  --reuse-dataset tops-450-20260918 \
  --models sam3 \
  --device mps \
  --download-weights \
  --splits dev \
  --limit 8 \
  --skip-encoder \
  --run-id sam3-mps-dev8
```

4. 통과하면 450장. 이미 있는 Grounded SAM 2.1 캐시(`sam-mps-450`)는 재사용하고 SAM3만 새로 돌린다.

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --config experiments/garment_segmentation/config.sam.yaml \
  --reuse-dataset tops-450-20260918 \
  --reuse-proposals sam-mps-450 \
  --models grounded_sam21 sam3 \
  --device mps \
  --skip-encoder \
  --run-id sam-with-sam3-450
```

첫 450장 다운로드가 아직이면 `--download-weights`를 붙인다. SAM 3는 약 848M이라 MPS 16GB에서도 여유가 없고, 장당 수 초~십수 초가 날 수 있다. OOM이면 `--device cpu`로 바꾸고 시간을 더 본다. 결과: `artifacts/garment_segmentation/<run_id>/report/index.html`

`owlv2_sam`은 `--models owlv2_sam`으로만 추가한다.

결과 리포트: `artifacts/garment_segmentation/<run_id>/report/index.html`

## 타깃 선택

모델 proposal과 resolver를 분리 평가한다.

- `max_area`
- `torso_prior`
- `metadata` (FashionSigLIP crop ↔ 상품명)
- `combined` (dev에서 동결한 가중치)

이미지별 수동 프롬프트 변경은 금지한다. proposal cache가 있으면 resolver만 다시 돌릴 수 있다.

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --reuse-dataset <freeze-run-id> \
  --reuse-proposals <proposal-run-id>
```

## 평가

- 자동: proposal 생성률, mask 수, 중복률, fragmentation, 면적 비율, 타깃 Top-1, target-absent FP, p50/p95 latency, peak memory, 실패율
- 육안: target_match, coverage, leakage, boundary, usable. 최소 50장은 두 명 독립 평가 후 weighted kappa
- primary/stress 분리, population weight와 source-balanced 수치, bootstrap 95% CI, paired 차이
- 상위 후보 차이가 5%p 이내이거나 경계 품질이 불안정하면 50~100장 pixel-gold 2차 평가를 권고

주석 HTML은 freeze 산출물의 `dataset/annotate.html`이다. 클릭 좌표는 정규화 [0, 1]로 저장한다.

```bash
./.venv/bin/python experiments/garment_segmentation/test_experiment.py
./.venv/bin/python experiments/garment_segmentation/run.py --models dummy --splits dev --limit 8 --skip-encoder
```

선정된 1~2개 모델의 raw/crop/hard-mask/soft-mask가 FashionSigLIP 정렬에 미치는 영향은 아래 전처리 비교 실험에서 다룬다.

## 전처리 A/B/C/D 임베딩 비교

작성일: 2026-09-18

같은 450장 상의에 대해 FashionSigLIP을 네 번 넣는다.

| 암 | 입력 |
| --- | --- |
| A | 원본 |
| B | `grounded_sam21` + `torso_prior` 박스를 12% 패딩한 crop |
| C | B와 같은 crop + SegFormer 상의 마스크 soft alpha + 원본 블러 배경 |
| D | B와 같은 crop + hard mask |

마스크는 `segformer_combined` 선택값을 그대로 쓰지 않는다. combined가 Background를 고르는 경우가 있어, **SAM 패딩 박스와 겹치는 SegFormer 상의 클래스(`Upper-clothes`, `Dress`) 합집합**을 쓴다. combined 선택은 상의이고 박스를 충분히 채울 때만 보조로 쓴다.

지금 있는 cache는 각 8장뿐이다. 450장 결론을 내려면 먼저 전체 SAM/SegFormer를 돌린다.

```bash
# 1) 박스: grounded_sam21 torso_prior
./.venv/bin/python experiments/garment_segmentation/run.py \
  --config experiments/garment_segmentation/config.sam.yaml \
  --reuse-dataset tops-450-20260918 \
  --models grounded_sam21 \
  --skip-encoder \
  --run-id sam-cpu-450

# 2) 마스크: segformer (resolver는 기록용, 실제 마스크는 proposal 전체에서 박스 내부 상의를 고른다)
./.venv/bin/python experiments/garment_segmentation/run.py \
  --reuse-dataset tops-450-20260918 \
  --models segformer \
  --skip-encoder \
  --run-id cpu-light-450
```

```bash
cp experiments/garment_segmentation/config.preprocess.example.yaml \
   experiments/garment_segmentation/config.preprocess.yaml
# box_run / mask_run 을 위 run id 로 수정
```

전처리 이미지만 먼저 확인:

```bash
./.venv/bin/python experiments/garment_segmentation/preprocess_run.py \
  --previews-only \
  --allow-partial \
  --run-id preprocess-abcd-previews
```

임베딩 + 산점도 + 호버 리포트:

```bash
./.venv/bin/python experiments/garment_segmentation/preprocess_run.py \
  --run-id preprocess-abcd-450
```

8장 스모크(현재 cache):

```bash
./.venv/bin/python experiments/garment_segmentation/test_preprocess.py
./.venv/bin/python experiments/garment_segmentation/preprocess_run.py \
  --box-run sam-cpu-dev8 \
  --mask-run cpu-light-dev8 \
  --allow-partial \
  --splits dev \
  --limit 8 \
  --previews-only \
  --run-id preprocess-abcd-dev8
```

결과: `artifacts/garment_segmentation/<run_id>/report/index.html`. 점 hover 시 A/B/C/D 네 입력이 같이 보인다. 수치 정의는 리포트 4절에 있다.

1차 판정: 속성 Precision@k와 intra-inter gap이 A 대비 오르면 개선. C가 이기고 D가 뒤지면 soft mask를 채택한다. B가 C를 이기면 마스크를 아직 쓰지 않는다.

## SAM3 A vs C 전처리 비교

작성일: 2026-09-21

SegFormer ABCD와 같은 FashionSigLIP 지표·UMAP·호버 리포트를 쓰되, 박스·마스크 모두 SAM3이고 암은 A(원본)와 C(패딩 박스 + soft mask + 블러 배경)만 비교한다. B/D는 생성하지 않는다. 타깃 박스는 `combined` resolver로 고른다.

| 항목 | 값 |
| --- | --- |
| 표본 | freeze `tops-450-20260918` |
| 박스 | `sam3` + `combined` |
| 마스크 | 같은 런의 SAM3 proposal. 라벨 `shirt`/`hoodie`/`sweater`/`blouse` 중 패딩 박스와 겹치는 합집합 |
| fallback | `combined` 선택이 상의이고 박스를 충분히 채울 때만, 아니면 박스 전체 |
| 설정 | `experiments/garment_segmentation/config.preprocess.sam3.yaml` |

### 1) Hugging Face 로그인 (처음 한 번)

`facebook/sam3`는 gated다. 웹에서 라이선스에 동의한 뒤:

```bash
./.venv/bin/hf auth login
```

가중치를 아직 안 받았으면 아래 SAM3 명령에 `--download-weights`를 붙인다.

### 2) SAM3 450장 추론

MPS에서 SAM3만 대략 50–70분이고, metadata용 FashionSigLIP이 모델당 약 10–20분 더 든다. `--skip-encoder`는 붙이지 않는다. 제안은 끝났을 때만 쓰이므로 중간에 끊으면 처음부터 다시 돈다. `sam3-mps-450`에 dataset 복사만 있고 `proposals/sam3/proposals.jsonl`이 없으면 이 run-id를 그대로 써도 된다.

```bash
./.venv/bin/python experiments/garment_segmentation/run.py \
  --config experiments/garment_segmentation/config.sam.yaml \
  --reuse-dataset tops-450-20260918 \
  --models sam3 \
  --device mps \
  --run-id sam3-mps-450
```

완료 확인:

```bash
test -f artifacts/garment_segmentation/sam3-mps-450/proposals/sam3/proposals.jsonl \
  && test -f artifacts/garment_segmentation/sam3-mps-450/selected/sam3/combined/assignments.jsonl \
  && echo ok
```

OOM이면 `--device cpu`로 바꾸고 시간을 더 본다.

### 3) A/C 임베딩 + 리포트

FashionSigLIP 두 번(A, C). 450장 기준 인코더 시간은 기기마다 다르다.

```bash
./.venv/bin/python experiments/garment_segmentation/preprocess_run.py \
  --config experiments/garment_segmentation/config.preprocess.sam3.yaml \
  --run-id preprocess-ac-sam3-450
```

HTML만 다시 그리려면:

```bash
./.venv/bin/python experiments/garment_segmentation/preprocess_run.py \
  --config experiments/garment_segmentation/config.preprocess.sam3.yaml \
  --report-only \
  --run-id preprocess-ac-sam3-450
```

결과:

- `artifacts/garment_segmentation/preprocess-ac-sam3-450/report/index.html`
- `artifacts/garment_segmentation/preprocess-ac-sam3-450/report/previews.html`
- `artifacts/garment_segmentation/preprocess-ac-sam3-450/metrics/summary.json`

1차 판정: C가 A보다 속성 P@k·intra-inter gap이 오르면 SAM3 soft mask를 채택한다. A가 이기면 원본을 유지한다.
