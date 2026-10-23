# 공간 풀링 집중도와 가림 강건성 연구

기존 DynaPool 논문의 후속 연구용 독립 프로젝트입니다. M5 MacBook Air의 VS Code에서 데이터 다운로드, 학습, 평가, 통계표, 논문용 그림까지 실행하도록 구성했습니다. 실제 성능 수치나 채택 가능성을 미리 보장하는 프로젝트는 아닙니다. 아직 실험하지 않은 결과를 채우는 코드는 포함하지 않았습니다.

추천 제목은 **Spatial Concentration and Occlusion Robustness of Global Pooling in Lightweight Image Classification**입니다. 한국어 제목은 **경량 이미지 분류에서 전역 풀링의 공간 집중도와 가림 강건성 분석**입니다. 상세한 가설과 논문 구성은 RESEARCH_PLAN_KO.md에 있습니다.

## 1 시작 순서

1. ZIP을 풀고 `pool-occlusion-study` 폴더를 바탕화면의 AI 폴더 등 원하는 작업 위치로 옮깁니다.
2. VS Code에서 **파일 → 폴더 열기**로 이 폴더 자체를 엽니다.
3. VS Code의 **터미널 → 새 터미널**을 엽니다. 터미널 위치가 `pool-occlusion-study`인지 확인합니다.
4. 아래 명령을 순서대로 실행합니다. 다른 연구의 가상환경이 활성화되어 있다면 먼저 `deactivate`를 실행합니다.

```bash
bash setup_macos.sh
source .venv/bin/activate
python study.py doctor
```

Python 3.12의 Apple Silicon arm64 버전을 권장합니다. 설치 스크립트는 Python 3.11–3.13을 허용하고 Rosetta x86_64 Python을 거부합니다. 기존 dehaze 연구의 가상환경을 재사용하거나 수정할 필요가 없습니다. `POOLSTUDY_PYTHON=/원하는/python3 bash setup_macos.sh`로 인터프리터를 지정할 수도 있습니다.

VS Code에서 `Command + Shift + P` → **Python: Select Interpreter** → 이 폴더의 `.venv/bin/python`을 선택합니다. Microsoft의 Python 확장이 필요합니다.

`doctor` 출력에 `device: mps`, `mps_available: true`가 나오면 정상입니다. M5에서도 PyTorch의 장치명은 `mps`입니다. CUDA를 설치하지 않습니다. 본 실험은 FP32를 사용하며 AMP, torch.compile, MPS CPU fallback을 켜지 않습니다.

## 2 학습 전에 작은 실행 점검

```bash
python -m pytest -q tests
python study.py all --config configs/smoke.json
```

`smoke`는 무작위 이미지와 무작위 라벨로 전체 파이프라인만 검사합니다. CIFAR 다운로드가 필요하지 않습니다. 여기서 나온 정확도, CSV, 그림은 **논문에 쓰시면 안 됩니다**. 결과 폴더와 그림에 SYNTHETIC 표시가 붙습니다.

```text
outputs/smoke_SYNTHETIC_NOT_FOR_PAPER/
```

## 3 실제 데이터 다운로드와 예상 시간 확인

```bash
python study.py download --config configs/main.json
python study.py benchmark --config configs/main.json --batches 30
```

공식 CIFAR-100 Python 데이터가 자동으로 내려받아지고 torchvision이 파일 무결성을 검사합니다. 압축 파일은 약 161 MB이며, 데이터는 `data/`에 저장됩니다. 다운로드 실패 시 인터넷과 인증서를 확인한 뒤 다시 실행합니다. SSL 검증을 끄지 않습니다.

`benchmark`는 실제 CIFAR 이미지, crop/flip, 데이터 로딩, 역전파, 검증을 포함해 측정합니다. 5개 준비 배치 이후의 시간을 사용합니다. `outputs/main_cifar100_r20_timing/budget.csv`에 방법별 측정치와 전체 학습 예상 시간을 저장합니다. 이 측정은 합성 속도 점검과 다릅니다. 최종 가림 평가, 체크포인트 저장, 장시간 발열에 따른 감속은 추정치에 추가됩니다. 숫자를 측정하기 전에는 소요 시간을 확정하지 마십시오.

M5 24 GB에서는 모델보다 전체 실험 횟수가 시간을 결정합니다. 기본 설정은 **7개 방법 × 5개 시드 × 80 epoch = 35회 학습**입니다. 짧은 실습이 아니라 여러 번 나누어 돌리는 논문 실험입니다. 모든 학습은 순차적으로 진행합니다. 본 실험은 대형 사전학습 모델이나 ImageNet 다운로드를 요구하지 않습니다.

## 4 본 실험을 한 명령으로 실행

```bash
bash run_main.sh
```

가상환경을 만든 뒤 실행하는 명령입니다. macOS의 `caffeinate`로 유휴 절전을 막으며 아래 작업을 순서대로 진행합니다.

1. 학습용 CIFAR 데이터 확인 및 다운로드
2. 학습 데이터 45,000장과 검증 데이터 5,000장 분리
3. 모든 방법과 시드의 80 epoch 학습
4. 최종 epoch 체크포인트로 공식 test 10,000장 평가
5. 가림과 Gaussian noise 평가
6. 시드별 통계, CSV, LaTeX 표, PNG와 PDF 그림 생성

노트북은 전원에 연결하고 통풍되는 곳에서 사용하십시오. `caffeinate`는 덮개를 닫았을 때 학습 지속을 보장하지 않으므로 덮개는 열어 두십시오.

직접 실행하려면 다음 명령을 사용합니다.

```bash
caffeinate -i python study.py all --config configs/main.json
```

단계별로 나눌 수도 있습니다.

```bash
python study.py train --config configs/main.json
python study.py evaluate --config configs/main.json
python study.py report --config configs/main.json
```

평가는 모든 방법의 학습이 끝나야 시작됩니다. 학습 중에는 공식 test를 읽지 않고, 학습 데이터에서 떼어낸 검증셋만 봅니다. 체크포인트는 사전에 정한 **최종 epoch**이며 검증셋 또는 test 최고점을 골라 보고하지 않습니다. `download`는 test 파일도 내려받지만 모델 선택에 사용하지 않습니다.

## 5 중단하고 다시 실행

터미널에서 `Control + C`로 중단한 뒤 동일한 명령을 다시 실행합니다. 매 epoch 체크포인트를 저장하며, 완료된 방법과 시드는 건너뜁니다. 중단된 epoch는 처음부터 다시 계산합니다. 학습 데이터 순서와 증강은 시드·epoch·원본 이미지 ID로 결정되어 재시작에 영향을 덜 받습니다. MPS 자체의 비결정성 때문에 비트 단위 동일성까지 보장하지는 않습니다.

같은 설정을 다른 터미널에서 동시에 실행하면 PID 잠금으로 막습니다. 강제 종료 후 남은 잠금은 해당 프로세스가 없을 때 자동으로 정리합니다.

한 결과 폴더 안에서 설정 또는 모델 코드를 바꾸면 서로 다른 실험이 섞이지 않도록 중지합니다. 변경 실험은 JSON을 복사하고 `name`도 바꾸십시오. `epochs`, 배치 크기, 시드, device를 바꾸는 경우도 같습니다. 이미 평가한 test 결과를 보고 학습 설정을 반복 조정하면 독립 평가가 아니므로 논문에 그 사실을 밝혀야 합니다.

## 6 무엇을 비교하는가

모든 모델은 동일한 CIFAR ResNet-20과 선형 분류기를 사용합니다. 32×32 입력이 64×8×8 특징 맵으로 변환됩니다. ResNet-20은 약 0.28M 파라미터이며 CIFAR용 projection shortcut을 사용합니다. 기존 논문의 torchvision small-image ResNet-18과 동일한 모델은 아닙니다.

| 코드 이름 | 비교 방법 | 목적 |
|---|---|---|
| gap | Global Average Pooling | 균일한 공간 평균 |
| max | Global Max Pooling | 채널별 최댓값 선택 |
| gem | GeM | 학습 가능한 지수 p, 초기값 3 |
| attention | 1×1 score + spatial softmax | 기본 학습형 공간 가중치 |
| attention_entropy | Attention + 정규화된 KL to uniform | 기존 엔트로피 최대화 계열 대조군 |
| attention_floor | Attention + 엔트로피 하한 soft penalty | 일정 수준 미만의 집중에만 벌점 |
| mixed_attention | 0.5 Average + 0.5 Attention | 고정 혼합만으로 충분한지 확인 |

같은 시드에서 backbone과 classifier의 **초기 가중치가 정확히 같습니다**. Attention 계열의 score 초기값도 같습니다. 방법별 파라미터 개수가 데이터 순서나 증강 난수를 바꾸지 않습니다. 모든 방법에 SGD, momentum 0.9, weight decay 0.0005, 기본 LR 0.1, 5 epoch warmup, cosine decay를 동일하게 적용합니다. 사전학습, Mixup, CutMix, RandAugment는 사용하지 않습니다.

기존 논문의 4개 branch gate가 Attention branch를 고르는 현상과, 이 연구의 공간 softmax가 한 위치에 집중하는 현상은 **서로 다른 현상**입니다. 전자의 결과만으로 후자를 이미 입증했다고 쓰면 안 됩니다.

Attention spatial weights를 a라고 할 때 정규화된 엔트로피는 `h = -sum(a * log(a)) / log(64)`입니다.

```text
attention_entropy: L = CE + 0.05 × mean(1 - h)
attention_floor:   L = CE + 1.0 × mean(max(0, 0.8 - h)^2)
```

하한 방식은 제약 최적화가 아니라 soft penalty입니다. 실제 엔트로피가 항상 0.8 이상이라는 보장은 없습니다. 학습 내내 엔트로피가 높으면 벌점이 0이어서 일반 Attention과 동일하게 학습될 수도 있습니다. `history.csv`의 `train_penalty`, `train_entropy`, `fraction_below_floor`를 반드시 확인하십시오.

이 penalty의 성능 향상은 검증할 가설입니다. 결과가 같거나 나빠도 숨기지 말고 집중도와 강건성의 관계를 분석하십시오. Attention 엔트로피 정규화는 선행연구가 있으므로 최초 기법이라고 주장하면 안 됩니다. 이 코드는 AEM 원 논문의 전체 재현이 아니라 엔트로피 최대화 계열의 단순 대조군을 구현합니다.

## 7 평가 조건과 통계

학습은 clean crop/flip만 사용합니다. 기본 test 조건은 총 16개입니다.

| 조건 | 구성 |
|---|---|
| clean | 원본 이미지 1개 조건 |
| 사각 가림 | 요청 면적 10%, 25%, 40% × 평균색/무작위색 채움 × 2회 위치 반복 = 12개 |
| Gaussian noise | 픽셀 범위 [0,1]에서 sigma 0.03, 0.08, 0.15 = 3개 |

사각형은 이미지 경계 안에 완전히 들어갑니다. 변 길이를 정수로 반올림하므로 실제 면적은 각각 **9.765625%, 25%, 39.0625%**입니다. CSV와 그래프는 실제 면적을 기록합니다. 채움 종류와 학습 시드가 달라도 같은 원본 ID·반복 번호에는 동일한 위치를 사용합니다. 다른 면적에서 완전히 포함되는 nested mask를 보장하지 않습니다.

주 평가 지표는 **가림 12개 조건의 정확도를 시드 안에서 평균한 값**입니다. clean 정확도, 감소 폭, clean 정답 이미지에서의 정답 유지율, normalized occlusion AUC, noise 정확도, NLL, ECE, Brier는 보조 지표입니다.

평균 ± 표본 표준편차와 t 기반 95% 신뢰구간을 시드 단위로 계산합니다. 하한 방식과 Attention, entropy, 고정 혼합의 3개 주 비교에 Holm 보정을 적용합니다. clean 비교는 별도 보조 비교군입니다. 신뢰구간은 다중 비교 보정 전 구간입니다. 5개 시드도 검정력이 충분하다고 단정할 수 없습니다. 보조 실험의 3개 시드는 더 조심해서 해석하십시오. 여러 마스크나 10,000개 이미지를 독립된 학습 실행으로 취급하지 않습니다.

집중도와 정답 유지율의 Spearman 상관은 시드별 기술통계입니다. 클래스, 난이도, clean confidence가 영향을 줄 수 있으므로 인과관계로 쓰지 않습니다. Max와 GeM은 채널 전체에 공유되는 하나의 공간 확률 맵이 없어 entropy를 공란으로 둡니다. 임의의 heatmap을 만들어 비교하지 않습니다.

## 8 자동 생성되는 파일

```text
outputs/main_cifar100_r20/
  protocol.json                   전체 고정 설정
  source_sha256.json              실행 코드 해시
  environment-freeze.txt          실제 설치 버전
  split_indices.npz               고정 train/validation ID
  split_manifest.csv              분할과 라벨
  classes.json                    클래스 이름
  test_protocol.json              평가 조건과 지표
  runs/방법_seed번호/
    last.pt                       모델과 optimizer 체크포인트
    history.csv                   epoch별 학습과 검증 기록
    environment.json              장치와 실행 환경
    complexity.json               파라미터 수와 지원 모듈 MAC
    latency_raw.csv                첫 번째 시드에서 측정한 모든 시간
    test/
      metrics.csv                 조건별 정확도와 집중도
      clean.npz                   float32 확률과 정답 및 원본 ID
      조건이름.npz                 동일 형식의 각 조건 예측
      조건이름_predictions.csv.gz  이미지별 예측과 집중도
      조건이름_reliability.csv     신뢰도 구간 통계
      confusion_clean.csv         클래스별 혼동행렬
      per_class_clean.csv         클래스별 정확도
  tables/
    paper_table.csv               논문용 요약표
    paper_table.tex               LaTeX 표
    all_condition_metrics.csv     모든 조건 원자료
    seed_endpoints.csv            시드별 주요 지표
    summary_mean_sd_ci.csv        평균과 SD 및 95% CI
    paired_seed_tests.csv         짝지은 차이와 Holm p값
    entropy_robustness_correlations.csv
    association_방법_s번호.csv.gz
    complexity.csv
    latency_summary.csv
  figures/
    architecture_entropy_floor.png/pdf
    learning_curves.png/pdf
    accuracy_comparison.png/pdf
    occlusion_curves.png/pdf
    gaussian_noise_secondary.png/pdf
    spatial_entropy.png/pdf
    entropy_distribution_clean.png/pdf
    training_entropy_diagnostics.png/pdf
    paired_effects.png/pdf
    reliability_clean.png/pdf
    confusion_attention.png/pdf
    confusion_attention_floor.png/pdf
    pooling_weights_조건.png/pdf
  images/조건/id_번호.png           사전에 정한 원본과 가림 예시
  RESULTS_README.md                결과 해석 주의사항
```

NPZ의 `probs`는 float32 전체 클래스 확률입니다. `labels`, `ids`, `entropy`, `effective_positions`, `peak`, `occluded_mass`, `actual_area`도 저장됩니다. 확률을 다시 사용해 지표를 감사할 수 있습니다. 35회 본 실험은 예측 파일만 수 GB가 생길 수 있으므로 여유 공간을 확보하십시오. 데이터셋과 결과는 Git에 올리지 않도록 기본 제외했습니다.

논문에는 PDF 벡터 그림을 우선 사용합니다. 그림의 오류 막대는 별도 표기된 paired effects를 제외하면 시드 간 SD입니다. heatmap은 실제 pooling weight이고 Grad-CAM이 아닙니다. test 예시 ID는 미리 고정되어 있으며 성공 사례만 골라 보여주지 않습니다. `occluded_mass`는 8×8 격자와 입력 가림의 기하학적 겹침이며, receptive field 전체나 인과적 설명을 나타내지 않습니다.

추론 시간은 첫 시드 모델에서 FP32, batch 1과 128, 30회 warmup, 100회 forward × 5 trials로 측정합니다. 매 호출을 synchronize하고 입력을 미리 장치에 올립니다. 데이터 로딩·전송·진단 지표 계산을 제외한 모델 forward 시간입니다. MAC은 Conv2d와 Linear만 계산하며 softmax, pooling reduction, BN, 활성화 등을 제외합니다. 이를 총 FLOPs라고 표현하면 안 됩니다. 장치·배치 크기와 함께 보고하십시오.

## 9 논문 완성도를 높이는 추가 실험

권장 최소 순서는 **본 실험 완료 → CIFAR-10 확인 실험**입니다. 한 데이터셋에서 나온 순위를 일반화하지 않습니다. CIFAR-10에서는 핵심 4개 방법 × 3개 시드를 재검증합니다. 추가로 시간 여유가 있을 때 다른 backbone과 학습 가림 대조군을 진행합니다.

```bash
python study.py all --config configs/confirm_cifar10.json
python study.py all --config configs/confirm_resnet32.json
python study.py all --config configs/train_occlusion_control.json
```

각 줄은 별개의 실험입니다. 세 줄 모두 자동으로 실행할 필요는 없습니다. CIFAR-10 확인 실험만으로도 12회 학습이 추가됩니다. ResNet-32도 12회, 학습 가림 대조군은 9회가 추가됩니다. 여러 모델을 동시에 학습시키지 마십시오.

학습 가림 대조군은 crop/flip 후 50% 확률로 25% 면적의 평균색 사각형을 넣습니다. Cutout과 관련된 **명시적 가림 증강 대조군**이며 Cutout 원 논문의 경계 처리와 동일한 재현은 아닙니다. 모든 방법에 같은 증강을 적용한 뒤 정규화 효과가 남는지 봅니다.

## 10 하이퍼파라미터 민감도

```bash
python sensitivity.py --config configs/main.json --epochs 40
```

학습 데이터에서 분리한 validation만 사용해 다음 8개 개발 설정을 각각 첫 시드로 학습·평가합니다. 공식 test는 읽지 않습니다.

- floor threshold 0.6, 0.8, 0.9와 lambda 1.0
- threshold 0.8에서 floor lambda 0.2, 5.0
- entropy lambda 0.01, 0.05, 0.2

`outputs/development_summary_cifar100_resnet20_e40/`에 CSV와 그림이 생깁니다. 개발 실험은 본 실험과 별도 이름으로 저장하며 본 설정을 자동 수정하지 않습니다. 민감도 결과로 설정을 선택하려면 **공식 test 평가 전에** 새 config 이름으로 그 선택을 고정하십시오. 이미 test를 봤다면 사후 탐색으로 명시해야 합니다. 1개 시드·40 epoch의 민감도는 최종 성능 검증을 대신하지 않습니다.

## 11 빠른 실제 데이터 확인

```bash
python study.py all --config configs/pilot.json
```

실제 CIFAR-100에서 3개 방법을 1개 시드·10 epoch로 학습합니다. 학습이 정상 진행되는지와 loss 크기를 확인하는 용도입니다. pilot 결과도 충분한 논문 실험은 아닙니다. 전체 반복과 다른 데이터셋 확인을 생략하고 pilot만 제출하지 마십시오.

## 12 자주 생기는 문제

- `ModuleNotFoundError`: `.venv` 활성화와 VS Code 인터프리터를 확인하고 `python -m pip install -r requirements.txt`를 실행합니다.
- `MPS unavailable`: `python -c 'import platform; print(platform.machine())'`가 arm64인지 확인합니다. `python study.py doctor --device cpu`는 CPU 검사일 뿐 M5 GPU 사용 확인이 아닙니다.
- 메모리 오류: 실행 중인 다른 학습을 종료합니다. 필요하면 config를 복사하고 새 `name`에서 batch_size 64, eval_batch_size 128로 모든 방법을 동일하게 실행합니다.
- 설정이나 코드 변경 오류: 기존 결과에 덮어쓰지 말고 새 이름으로 시작합니다. 기존 결과를 지울 필요는 없습니다.
- `Attention`과 `attention_floor`가 같음: 벌점이 0인지 확인합니다. 효과가 없는 결과도 연구 결과입니다. test를 보고 threshold를 계속 바꾸지 마십시오.
- 그래프만 다시 만들기: `python study.py report --config configs/main.json`을 실행합니다. 학습과 평가는 반복하지 않습니다. 코드 수정이 있으면 해시 검사가 막을 수 있으므로 다른 구현 결과를 섞지 마십시오.
- 네트워크 오류: 설치와 데이터 다운로드에는 인터넷이 필요합니다. 데이터와 패키지를 받은 뒤 학습·평가는 오프라인에서 가능합니다.

## 13 제출 전 확인할 사항

`RESEARCH_PLAN_KO.md`의 기여 범위와 `VALIDATION.md`의 검증 범위를 읽으십시오. 학습 곡선의 수렴, 모든 시드 완료, clean 손실과 강건성 간 절충, entropy penalty 활성 여부, 사전 고정 설정, 비교군, 통계, 선행연구 중복을 확인한 뒤 본문을 작성합니다. 결과를 먼저 확인하지 않고 성능 우위를 전제로 제목·초록·결론을 확정하지 마십시오.
