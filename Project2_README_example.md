# 이미지 분류 학습 전략 비교와 오류 분석

동일한 ResNet18의 Training from Scratch와 Transfer Learning을 비교한 결과, Partial Fine-tuning에서 Validation Macro F1은 0.770에서 0.860으로 높아졌고, 1회 학습·검증 시간은 10분에서 6분으로 단축되었습니다. 오분류 이미지와 VLM 분석을 바탕으로 밝기 증강 실험을 수행했으나 검증 Macro F1이 낮아져, 밝기 증강을 적용하지 않은 Partial Fine-tuning을 최종 전략으로 선정했습니다.

> [Dataset 이름]을 활용하여 이미지 분류 모델의 학습·평가와 OpenAI API를 이용한 VLM 분석을 진행했습니다.

*작성 방법을 보여주기 위한 예시입니다. 데이터 구성, 분석 내용, 성능과 시간은 가상이며, 실제 작성 시 직접 수행한 내용과 결과로 바꿉니다. 이미지 경로 역시 첨부 방법을 보여주기 위한 예시이며 실제 이미지 파일은 제공하지 않습니다.*

## 1. 프로젝트 개요

- **해결하려던 문제:** 정상과 비정상 이미지를 구분하는 분류 문제입니다. 주어진 학습 시간 안에서 분류 성능과 학습 효율을 고려하여 적합한 학습 전략을 선정하고자 했습니다. Training from Scratch와 Transfer Learning을 비교하고, Transfer Learning 전략으로 Feature Extraction, Partial Fine-tuning, Full Fine-tuning을 검토했습니다. 이 예시에서는 Partial Fine-tuning을 선택하여 실제 비교 실험을 수행했습니다.
- **목표와 평가 기준:** Training from Scratch와 Transfer Learning의 분류 성능과 학습 시간을 비교했습니다. 프로젝트 초기에 Macro F1을 우선 평가 지표로 정하고, 정상·비정상 클래스별 성능과 오분류 사례를 함께 확인했습니다.

| 항목 | 내용 |
|---|---|
| 기간 | 8일 |
| 사용 데이터 | [Dataset 이름] ([데이터셋 출처 링크]) |
| 주요 기술 | Python, PyTorch, torchvision, scikit-learn, Matplotlib, uv, Jupyter Notebook, OpenAI API |

## 2. 프로젝트 구조

프로젝트 디렉터리 구조는 다음과 같습니다.

```text
project/
├── readme.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── data/
│   └── raw/
│       └── images/
│           ├── class_a/          # 정상
│           └── class_b/          # 비정상
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── models/
│   └── best_model.pth
└── figures/
    ├── learning_curves.png
    ├── confusion_matrix.png
    └── error_examples.png
```

## 3. 설치 및 실행 방법

### 환경 설치

Python 버전은 `.python-version`, 패키지와 버전은 `pyproject.toml` 및 `uv.lock`을 기준으로 합니다. Notebook 실행에 필요한 `ipykernel`도 프로젝트 의존성에 포함합니다.

[uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 uv를 설치한 뒤, 실제 저장소 주소와 폴더명을 사용해 실행합니다.

```bash
git clone <repository-url>
cd <repository-name>
uv sync
```

### 데이터 준비 및 실행

1. 정상 이미지는 `data/raw/images/class_a/`, 비정상 이미지는 `data/raw/images/class_b/`에 준비합니다. 클래스 매핑은 정상 0, 비정상 1로 설정했습니다.
2. VS Code에서 프로젝트 폴더와 Notebook을 열고, **Select Kernel**에서 프로젝트 **`.venv`의 Python 환경**을 선택합니다.
3. 아래 순서대로 Notebook을 실행합니다.

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | 이미지 구성·품질 확인과 학습 데이터 분석 |
| 2 | `notebooks/02_preprocessing.ipynb` | 동일한 분할 기준 적용, 이미지 변환 및 DataLoader 구성 |
| 3 | `notebooks/03_modeling.ipynb` | 학습·튜닝·오류·VLM 분석, 개선 실험 비교와 최종 평가 |

분할 기준과 이미지 목록, 시드는 두 데이터 처리 Notebook에서 동일하게 유지했습니다. 실행 GPU, 라이브러리 버전, 가중치 식별자와 상세 학습 설정은 `03_modeling.ipynb`에 기록했습니다.

최종 가중치는 `models/best_model.pth`에 저장했습니다. OpenAI API를 통한 VLM 분석에 사용한 모델 ID·프롬프트와 실행에 필요한 환경 변수명은 `03_modeling.ipynb`의 설정 부분에 기록했습니다.

## 4. 수행 내역

직접 수행한 작업과 주요 판단을 다음과 같이 정리했습니다.

| 작업 구분 | 직접 수행한 작업 | 근거 자료 |
|---|---|---|
| 데이터 분석 및 전처리 | 정상·비정상 이미지의 분포와 품질을 확인하고, 데이터 분할·이미지 변환·DataLoader를 구성했습니다. | `notebooks/01_eda.ipynb`, `notebooks/02_preprocessing.ipynb` |
| 모델 학습 및 비교 | ResNet18의 Training from Scratch와 Partial Fine-tuning을 수행하고, 학습률 조정에 따른 성능과 학습 시간을 비교했습니다. | `notebooks/03_modeling.ipynb` — 모델 학습 및 튜닝 |
| 오류·VLM 분석 | 검증 데이터에서 오분류 10장과 올바른 예측 4장을 선정했습니다. 실제 이미지와 OpenAI API를 통한 VLM 설명을 대조하여 밝기·대비와 시각적 특징에 관한 개선 가설을 정리했습니다. | `notebooks/03_modeling.ipynb` — 오류·VLM 분석 |
| 개선 실험 및 검토 | 학습 이미지에 밝기 증강을 추가하고 전체·클래스별 성능 변화를 비교했습니다. 비정상 Recall은 높아졌지만 정상 Recall과 Macro F1이 낮아져 기존 모델을 유지했습니다. | `notebooks/03_modeling.ipynb` — 밝기 증강 실험 |
| 결과 정리 | Validation 결과와 최종 Test 결과를 구분하고, 모델 선정 이유·남은 오류·실행 방법을 문서화했습니다. | `notebooks/03_modeling.ipynb` — 최종 평가, `readme.md` |

## 5. 문제 해결 과정

### 주요 접근

- **데이터:** 정상 이미지 1,000장과 비정상 이미지 1,000장으로 구성된 RGB 이미지 2,000장을 사용했습니다. 정답은 데이터셋에서 제공한 라벨을 기준으로 했습니다. Train / Validation / Test를 1,200 / 400 / 400장으로 나누고, 각 분할의 정상·비정상 비율을 동일하게 유지했습니다. 동일 대상·유사 이미지가 서로 다른 분할에 섞이지 않도록 확인했으며, 상세 EDA는 Train 데이터에서 수행했습니다.
- **학습 방법:** 입력 크기를 224×224로 맞추고, 무작위 초기화한 ResNet18과 사전학습 가중치를 활용한 ResNet18을 비교했습니다. Transfer Learning에서는 사전학습 특징을 활용하면서 현재 데이터에 맞게 일부 특징을 조정하기 위해 Partial Fine-tuning을 선택했습니다. `layer4`와 `fc`를 학습하고 나머지 레이어는 고정했습니다.
- **검증:** 동일한 Validation 데이터에서 학습 전략과 설정을 비교하고 최종 모델을 선정했습니다. Test 데이터는 모델 선정이 완료된 뒤 최종 평가에 사용했습니다. 전체 설정과 학습 곡선은 Notebook에 기록했습니다.

### 핵심 판단: VLM의 설명을 개선 가설로 검토

검증 오류를 살펴보니 일부 비정상 이미지가 정상으로 분류된 사례에서 밝기가 낮고 대상과 배경의 경계가 뚜렷하지 않았습니다. 오분류 10장과 올바르게 분류한 4장을 선정하여 OpenAI API를 통해 이미지의 시각적 특징을 추가로 분석했습니다.

VLM은 낮은 대비 때문에 정상·비정상을 구분하는 세부 특징이 충분히 드러나지 않을 가능성을 제시했습니다. 실제 이미지에서도 낮은 대비를 확인했지만, 이것이 모델의 오분류 원인이라고 확정하지는 않았습니다.

이를 바탕으로 **학습 이미지의 밝기 다양성을 늘리면 비정상 이미지를 놓치는 오류가 줄어들 수 있다**는 가설을 수립했습니다. 기존 모델 설정을 유지한 채 Train 이미지에만 ±20% 밝기 증강을 추가하고, 변경 전후의 전체·클래스별 결과를 비교했습니다. Validation / Test 전처리는 변경하지 않았습니다.

사용한 VLM 모델 ID, 프롬프트, 원문 응답과 이미지 ID는 `notebooks/03_modeling.ipynb`의 오류 분석 부분에 기록했습니다. VLM의 설명은 개선 가설을 검토하는 데 활용했으며, 이미지의 정답 라벨이나 확정적인 오류 원인으로 사용하지 않았습니다.

## 6. 주요 실험 및 결과

기본 학습과 학습률 조정 등 전체 실험 이력은 Notebook에 기록하고, 아래에는 최종 모델 선정에 사용한 대표 결과를 정리했습니다. 학습 전략과 개선 실험은 동일한 Validation 데이터에서 비교했습니다.

### 학습 전략 비교

각 학습 전략에서 학습률 조정 후 선정한 모델을 비교했습니다.

| 비교 항목 | Training from Scratch | Partial Fine-tuning |
|---|---|---|
| 모델 구조 | ResNet18 | ResNet18 |
| 초기 가중치 | 무작위 초기화 | ImageNet 사전학습 가중치 |
| 학습 범위 | 전체 레이어 | `layer4`, `fc` |
| 전체 파라미터 수 | 11,177,538 | 11,177,538 |
| 학습 가능 파라미터 수 | 11,177,538 | 8,394,754 |
| 학습 Epoch 수 | 20 | 20 |
| 학습 시간 | 10분 | 6분 |
| Validation Macro F1 | 0.770 | 0.860 |

*파라미터 수는 torchvision의 표준 ResNet18에서 최종 분류 레이어의 출력 크기를 2로 변경하고, 위 학습 범위를 적용한 경우를 기준으로 작성했습니다.*

Partial Fine-tuning은 전체 파라미터 수는 동일하지만 학습 가능한 파라미터 수가 더 적었습니다. 이번 실험 조건에서는 Scratch보다 높은 검증 성능과 짧은 학습 시간을 보였습니다.

학습 시간은 동일한 GPU 환경에서 각 모델을 20 Epoch 학습하고 검증하는 1회 실행에 소요된 시간이며, 전체 튜닝 시간은 포함하지 않았습니다. 초기 가중치·학습 범위·학습률이 달라 결과 차이를 한 가지 요인의 효과로 단정하지 않았습니다. 전체 학습 설정, Epoch별 시간, 학습 곡선과 클래스별 평가는 `notebooks/03_modeling.ipynb`와 `figures/`에 기록했습니다.

### 주요 실험 및 개선 결과

| 시도 또는 방법 | 목적·변경 내용 | 검증 결과 | 해석 및 결정 |
|---|---|---|---|
| Training from Scratch — 기본 학습 | 무작위 초기화한 ResNet18의 전체 레이어를 학습했습니다. | Macro F1 0.730 | Scratch의 기본 성능을 확인했습니다. |
| Training from Scratch — 튜닝 | 기본 학습 설정에서 학습률을 조정했습니다. | Macro F1 0.770 / 10분 | 기본 학습보다 성능이 높아 Scratch의 대표 결과로 선정했습니다. |
| Partial Fine-tuning — 기본 학습 | 사전학습 가중치를 활용하여 `layer4`와 `fc`를 학습했습니다. | Macro F1 0.840 | Transfer Learning의 기본 성능을 확인했습니다. |
| Partial Fine-tuning — 튜닝 | 기본 학습 설정에서 학습률을 조정했습니다. | Macro F1 0.860 / 6분 | 비교한 조건에서 검증 Macro F1이 가장 높고 학습 시간이 짧아 최종 후보로 선택했습니다. |
| Partial Fine-tuning + 밝기 증강 | 오류 분석에서 수립한 가설에 따라 최종 후보 모델의 학습 설정을 유지하고 Train 이미지의 밝기 증강만 추가했습니다. | Macro F1 0.850 / 6.5분 | 우선 평가 지표인 Macro F1이 낮아져 채택하지 않고 기존 모델을 유지했습니다. |

밝기 증강 후 비정상 Recall은 0.820→0.840으로 높아졌지만, 정상 Recall은 0.900→0.860으로 낮아졌습니다. 이번 프로젝트에서는 사전에 정한 Macro F1을 우선 기준으로 적용했으므로, 비정상 Recall의 개선만으로 해당 방법을 채택하지 않았습니다.

### 최종 결과 및 해석

- **최종 결과:** 밝기 증강을 적용하지 않은 Partial Fine-tuning 모델을 선정했습니다. 이후 Test 400장에서 Accuracy 0.850, Macro F1 약 0.850을 확인했습니다.
- **결과의 의미:** 이번 데이터와 실험 조건에서는 Partial Fine-tuning이 Training from Scratch보다 높은 검증 성능과 짧은 학습 시간을 보였습니다. 오류 분석에서 제안한 방법도 전체 평가 결과를 확인한 뒤 적용 여부를 결정했습니다.
- **근거 자료:** `notebooks/03_modeling.ipynb`의 학습 전략 비교, 개선 실험 및 최종 평가 부분과 `figures/`의 시각화 결과를 참고합니다.

### 주요 결과 시각화

*아래 이미지 경로는 README 작성 예시입니다. 실제 프로젝트에서는 분석 코드로 생성한 이미지를 해당 위치에 저장하고, 실제 결과에 맞게 설명을 작성합니다.*

#### 학습 곡선

![Learning Curves](figures/learning_curves.png)

Training from Scratch와 Partial Fine-tuning의 Epoch별 Train / Validation Loss 및 Validation Macro F1을 비교합니다. 학습 곡선을 바탕으로 수렴 속도와 과적합 여부를 확인합니다.

#### 최종 모델 평가

![Confusion Matrix](figures/confusion_matrix.png)

최종 Partial Fine-tuning 모델의 Test 데이터에 대한 Confusion Matrix입니다. 전체 Accuracy와 Macro F1뿐만 아니라 정상·비정상 클래스별 오류를 함께 확인합니다.

#### 대표 오분류 사례

![Error Examples](figures/error_examples.png)

Validation 데이터의 대표 오분류 이미지와 정답·예측 클래스를 함께 표시합니다. 일부 비정상 이미지에서 낮은 밝기와 대비를 관찰했으며, VLM의 설명을 참고하여 밝기 증강에 관한 개선 가설을 수립했습니다.

## 7. 한계 및 배운 점

- **한계:** 단일 데이터 분할과 시드에서 확인한 결과이며, 실제 운영 데이터에서는 검증하지 못했습니다. VLM으로 분석한 14장도 전체 오분류를 대표하지 않으므로, 관찰한 특징을 모든 오류에 일반화하지 않았습니다.
- **배운 점:** VLM이 제시한 설명을 개선 가설로 정리하고 실제 실험으로 확인하는 과정이 필요했습니다. 한 클래스의 성능이 좋아지더라도 다른 클래스와 전체 지표가 나빠질 수 있어, 처음 설정한 평가 기준에 따라 적용 여부를 결정해야 한다는 점을 확인했습니다.
- **다음 접근:** 밝기 구간별 오류 비율을 더 많은 사례에서 확인하고, Train / Validation 데이터에서 증강 강도를 비교할 계획입니다. 선택한 방법은 별도의 평가 데이터에서 다시 검증할 계획입니다.
