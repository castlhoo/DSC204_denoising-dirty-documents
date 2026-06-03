# 더러운 문서 노이즈 제거 (Denoising Dirty Documents)

스캔된 문서 이미지에서 노이즈(커피 얼룩, 변색, 불균일한 조명, 구겨짐 등)를 제거하여
OCR에 적합한 깨끗한 이미지를 복원하는 프로젝트입니다.
[AI 코딩 짐](https://aicodinggym.com)의 **MLE-bench** 챌린지(캐글 *Denoising Dirty Documents*)로 진행되었습니다.

| 항목 | 값 |
|---|---|
| **최종 점수** | **0.01514 RMSE** |
| **순위** | **5 / 162 (상위 3.1%) — 골드 등급** |
| AI 베이스라인 | 0.01649 RMSE |
| 개선폭 | AI 대비 **+8.2% (상대)** |
| 모델 | ImprovedUNet (기본 채널 32, 약 195만 파라미터) |

> 실험 과정·의사결정의 상세 분석은 [REPORT_README.md](REPORT_README.md)에 정리되어 있습니다.
> 이 문서는 프로젝트 전체 구조와 실행 방법을 빠르게 파악하기 위한 진입점입니다.

---

## 1. 문제 개요

- **입력:** 노이즈가 있는 그레이스케일 문서 이미지 (PNG)
- **출력:** 각 픽셀의 복원된 밝기값을 담은 CSV (`0.0` = 검정, `1.0` = 흰색)
- **평가 지표:** 예측 픽셀과 정답 픽셀 간 **RMSE** (낮을수록 좋음)
- **데이터:** 학습 쌍 115장(노이즈 + 정답), 테스트 29장(노이즈만)

제출 CSV 형식 (픽셀 단위로 펼침):

```
id,value
1_1_1,1.000000      # 이미지 1, 1행, 1열
1_2_1,0.984314
...
```

---

## 2. 접근 방법 요약

기존 AI 에이전트(`ResidualUNet`, 기본 채널 16, 2 에폭에서 조기 종료)의 5가지 약점을 분석하고
도메인 특화 개선을 적용했습니다.

| 영역 | 핵심 개선 | 이유 |
|---|---|---|
| **아키텍처** | 4단계 ImprovedUNet + 배치 정규화 | 더 큰 수용 영역 + 학습 안정화 (115장 소규모 데이터) |
| **전처리** | Otsu 배경 정규화 (직접 구현) | 문서 노이즈의 본질은 "불균일한 배경 밝기" → 배경을 ~1.0으로 정규화 |
| **출력** | 잔차 제한(±0.1) 제거, 직접 예측 | 얼룩이 심한 픽셀도 전체 범위로 보정 가능 |
| **학습** | 80 에폭, 인내값 15, 코사인 감쇠 학습률 | 조기 종료로 인한 미학습 방지 |
| **데이터** | 이미지당 패치 2→8, 밝기 흔들기 증강 | 에폭당 학습 다양성 4배 |
| **추론** | 테스트 시 증강(TTA, 4방향 뒤집기 평균) | 추가 학습 없이 검증 RMSE 약 17% 개선 |

검증된 결과만 채택했습니다 — **감마 후처리**(실행 4)와 **앙상블**(실행 5)은 모두 개선 효과가 없어
최종 제출에서 제외했습니다(자세한 내용은 보고서 4절 참조).

---

## 3. 디렉터리 구조

```
denoising-dirty-documents/
├── README.md                     # (이 문서) 프로젝트 개요·실행 가이드
├── REPORT_README.md              # 상세 분석 보고서 (실험 타임라인, 의사결정 근거)
│
├── train_sean.py                 # ⭐ 메인 학습/추론 파이프라인
├── postprocess_predict.py        # 감마 후처리 튜닝 실험 (개선 없음)
├── ensemble_predict.py           # 다중 모델 앙상블 추론 (개선 없음)
├── colab_train_ensemble.ipynb    # Colab GPU 앙상블 학습 노트북
│
├── unet_sean_best.pth            # ⭐ 최적 모델 가중치 (67 에폭, 검증 RMSE=0.01457)
│
├── predictions_sean.csv          # ⭐ 최종 제출본 (테스트 RMSE 0.01514)
├── predictions_postprocessed.csv # 감마 후처리 결과
├── predictions_ensemble.csv      # 앙상블 결과 (RMSE 0.01539, 더 나쁨)
│
├── data/
│   ├── description.md            # 대회 설명
│   ├── sampleSubmission.csv      # 제출 형식 + 픽셀 순서 정의
│   ├── train/                    # 노이즈 학습 이미지 115장
│   ├── train_cleaned/            # 정답(클린) 학습 이미지 115장
│   └── test/                     # 테스트 이미지 29장
│
├── .log/                         # 세션 로그 (의사결정 전체 기록)
├── AGENTS.md                     # AI 코딩 짐 챌린지 지침
└── CLAUDE.md / GEMINI.md         # 에이전트별 설정
```

⭐ = 최종 결과 재현에 필요한 핵심 파일

---

## 4. 핵심 모듈 설명

### `train_sean.py` — 메인 파이프라인
학습부터 추론·CSV 생성까지 모두 담당합니다. 다른 스크립트도 여기서 함수를 가져다 씁니다.

- `ImprovedUNet` — 4단계 U-Net (인코더 3단 + 병목 + 디코더 3단), 배치 정규화 포함
- `otsu_background_normalize()` — 히스토그램 기반 Otsu 임계값을 직접 계산해 배경 밝기로 나눔
- `RandomPatchDataset` — 랜덤 패치 추출 + 증강(좌우/상하 뒤집기, 90도 회전, 밝기 흔들기)
- `predict_single()` — 반사 패딩 → 추론 → 테스트 시 증강 4방향 평균
- `write_predictions_csv()` — 제출 형식(`id,value`)으로 픽셀 단위 출력

### `postprocess_predict.py` — 감마 후처리 (실험)
예측값에 `출력값^감마`를 적용해 배경↔텍스트 대비를 강화하려는 시도.
검증셋에서 감마 0.7~2.0을 탐색한 결과 **감마=1.0(변화 없음)이 최적** → 모델 출력이 이미 잘 보정됨을 확인.

### `ensemble_predict.py` — 앙상블 추론 (실험)
서로 다른 시드로 학습한 여러 체크포인트의 예측을 평균. 기본 채널 48 모델 3개 앙상블은
0.01539로 단일 모델(0.01514)보다 **나빴음** — 소규모 데이터에서 큰 모델의 과적합으로 추정.

### `colab_train_ensemble.ipynb` — GPU 학습 노트북
Colab T4 GPU에서 시드 42/123/456 세 모델을 학습하고 앙상블 예측을 생성하는 노트북.

---

## 5. 환경 설정

```bash
pip install torch torchvision numpy pillow
```

GPU(CUDA)가 있으면 자동으로 사용하며, 없으면 CPU로 동작합니다.
데이터가 없다면:

```bash
aicodinggym mle download denoising-dirty-documents
```

---

## 6. 최종 결과 재현

```bash
# 1) 모델 학습 (CPU: 약 3시간 / GPU: 약 20분)
python train_sean.py \
    --data-dir data \
    --epochs 80 \
    --batch-size 16 \
    --patches-per-image 8 \
    --patch-size 64 \
    --base-channels 32 \
    --patience 15 \
    --seed 42 \
    --model-path unet_sean_best.pth \
    --output-csv predictions_sean.csv

# 2) 제출
aicodinggym mle submit denoising-dirty-documents -F predictions_sean.csv
```

> 동봉된 `unet_sean_best.pth`로 학습을 건너뛰고 바로 추론만 하려면
> `postprocess_predict.py`(감마=1.0이면 후처리 없음)를 추론 용도로 활용할 수 있습니다.

예상 결과: **RMSE ≈ 0.015, 골드 등급, 상위 5%**.

기본 하이퍼파라미터는 `train_sean.py`의 `argparse` 정의를 참고하세요
(스크립트 기본값은 에폭=80, 패치=10, 인내값=15이며, 위 명령은 재현 보고서 기준 설정입니다).

---

## 7. 주요 하이퍼파라미터 (`train_sean.py`)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data-dir` | (필수) | `train/`, `train_cleaned/`, `test/`, `sampleSubmission.csv`가 있는 폴더 |
| `--epochs` | 80 | 최대 학습 에폭 |
| `--batch-size` | 16 | 배치 크기 |
| `--lr` | 1e-3 | 초기 학습률 (코사인 감쇠로 1e-6까지 감소) |
| `--patch-size` | 128 | 학습 패치 크기 |
| `--patches-per-image` | 10 | 이미지당 추출 패치 수 |
| `--val-split` | 0.1 | 검증 분할 비율 |
| `--patience` | 15 | 조기 종료 인내 에폭 |
| `--base-channels` | 32 | U-Net 기본 채널 수 (용량) |
| `--seed` | 42 | 랜덤 시드 |

---

## 8. 참고

- 상세 실험 분석·회고: [REPORT_README.md](REPORT_README.md)
- 대회 원문 설명: [data/description.md](data/description.md)
- 세션 로그: [.log/](.log/)
- 챌린지 지침: [AGENTS.md](AGENTS.md)
```
