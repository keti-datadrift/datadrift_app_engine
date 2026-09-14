# 의류 속성 임베딩 군집화 실험

작성일: 2026-09-11

로컬 retail pack(`config.yaml`의 `input_root`)에 있는 기존 WebP와 상품 메타만 읽어 Marqo FashionSigLIP 표현을 비교한다. 입력 이미지를 내려받거나 수정하지 않으며 모든 파생 산출물은 `artifacts/attribute_embedding/<run_id>`에 생성한다.

`config.yaml`은 gitignore 대상이다. `config.example.yaml`을 복사한 뒤 로컬 경로와 채널 폴더명(`disk_name`)을 채운다. 산출물·보고서에는 공개 별칭(`retail_a`, `retail_b`)만 남긴다.

## 실험군

- A: 전체 원본 FashionSigLIP 임베딩
- B: 원본 임베딩을 family 내부에서만 군집화
- C: train split의 family centroid를 제거한 residual 임베딩
- D: family-conditioned 색상·패턴·소재·형태 프롬프트 점수

같은 이미지와 고정 모델을 사용하면 배치 분리 여부는 원본 임베딩을 바꾸지 않는다. B는 표현 변경이 아니라 후보군·군집화 범위 변경 효과를 측정한다.

## 실행

프로젝트의 `.venv`만 사용한다.

```bash
cp experiments/attribute_embedding/config.example.yaml experiments/attribute_embedding/config.yaml
# config.yaml의 input_root / sources.*.disk_name 을 로컬 값으로 수정
./.venv/bin/python experiments/attribute_embedding/run.py
```

빠른 데이터 점검:

```bash
./.venv/bin/python experiments/attribute_embedding/run.py --inventory-only --sample-per-family 30
```

## 산출물

- `manifest.json`: 실행·모델·데이터 fingerprint
- `dataset/items.jsonl`, `dataset/inventory.json`: 고정 표본과 데이터 품질
- `representations/*.npz`: A~D 표현
- `clusters/assignments.jsonl`: 군집과 2D 좌표
- `metrics/summary.json`: 속성 이웃, category probe, 군집 지표
- `figures/`: 정적 지표 및 고정 query contact sheet
- `report/index.html`, `report/summary.md`: 비교 보고서

현재 속성 평가는 상품명에서 추출한 weak label 기반이다. 사람이 검수한 gold label을 만들기 전에는 실험군의 상대 비교를 예비 결과로만 해석한다.

## Family-local 차원압축 프로브

특정 family의 원본 `A_raw` 임베딩만 모아 PCA / SVD / UMAP / t-SNE를 비교한다.

```bash
./.venv/bin/python experiments/attribute_embedding/family_dr_probe.py --family tops
```

산출물: `artifacts/attribute_embedding/family-dr-<family>-<source-run>/index.html`
