# DynaPool 후속 연구 설계

## 추천 주제

**경량 이미지 분류에서 전역 풀링의 공간 집중도와 가림 강건성 분석**을 권합니다. 새 모델을 크게 설계하기보다, 기존 논문에서 다룬 전역 풀링을 유지하면서 평가 질문을 정확도에서 공간 가림에 대한 안정성까지 확장합니다.

영문 가제는 **Spatial Concentration and Occlusion Robustness of Global Pooling in Lightweight Image Classification**입니다. 실험 전에는 robustness improvement를 제목에 넣지 않습니다. 향상이 확인되지 않아도 연구 질문을 유지할 수 있는 제목입니다.

## 기존 논문과의 연결

첨부된 카메라레디 논문은 Tiny-ImageNet, small-image ResNet-18에서 Average, Max, GeM, Attention, DynaPool을 비교합니다. DynaPool gate의 Attention branch 편중, entropy-regularized gate와 고정 혼합 대조군을 이미 분석했고, 동적 혼합이 우월하다고 결론내리지 않습니다. 따라서 다음 논문에서 단순히 branch를 하나 더 붙이거나 동일 표를 CIFAR로 바꾸는 접근은 기여가 약합니다.

후속 질문은 **어떤 branch를 고르는가**에서 **공간 정보를 얼마나 넓게 모으며, 일부 픽셀이 가려졌을 때 무엇이 달라지는가**로 옮깁니다. 기존 gate의 확률 분포와 이번 spatial attention의 확률 분포는 다른 축입니다. 공간 집중도 문제는 새로 측정해야 하며 기존 결과로 입증된 것으로 쓰지 않습니다.

## 후보 비교

| 후보 | 구현 난이도 | 기존 논문과 연결 | 연구상 주의점 |
|---|---|---|---|
| 공간 집중도와 가림 강건성 | 낮음에서 중간 | 풀링과 entropy 분석을 직접 확장 | 합성 가림과 실세계 가림을 구분 |
| 학습 데이터 양에 따른 풀링 선택 | 낮음 | 동일 풀링 비교를 데이터 효율성으로 확장 | 데이터 비율마다 반복 학습이 늘고 기여가 작아질 수 있음 |
| GeM 지수와 noise 민감도 | 낮음 | 기존 논문의 GeM 결과를 확장 | 지수 변화의 효과와 backbone 학습 효과를 분리해야 함 |

동봉 코드는 첫 번째 주제를 완성형 파이프라인으로 구현합니다. 나머지 후보는 별도 논문을 위한 대안입니다.

## 연구 질문

1. clean 정확도에서 유리한 풀링이 합성 가림 상황에서도 유리한가?
2. 공간 Attention의 정규화된 entropy와 이미지별 가림 정답 유지율은 어떤 관계를 보이는가?
3. entropy를 항상 높이는 penalty와, 정해진 하한 미만에서만 작동하는 penalty는 clean 성능과 가림 강건성에서 어떤 절충을 보이는가?
4. 간단한 고정 Average + Attention 혼합으로 같은 효과를 얻을 수 있는가?
5. 다른 데이터셋 또는 같은 증강 조건에서도 관찰이 유지되는가?

첫 번째와 두 번째는 관찰 질문입니다. 세 번째와 네 번째는 통제된 학습 변경을 비교하지만, 각 모델의 backbone도 함께 학습되므로 pooling만의 직접 인과효과를 분리했다고 주장하지 않습니다.

## 수식

특징 맵의 위치를 i, 채널을 c로 두고, N = H × W = 64로 정의합니다.

```text
s_i = w^T X_i
a_i = exp(s_i) / sum_j exp(s_j)
z_c = sum_i a_i X_ci
h(a) = -sum_i a_i log(a_i) / log(N)

L_Attention = CE
L_Entropy   = CE + lambda_E × (1 - h(a))
L_Floor     = CE + lambda_F × max(0, h0 - h(a))^2

z_Mix = (1 - alpha) × GAP(X) + alpha × Attention(X)
```

기본값은 lambda_E = 0.05, lambda_F = 1.0, h0 = 0.8, alpha = 0.5입니다. 이 값은 사전에 정한 시작 설정이며 최적값이라고 주장하지 않습니다. soft penalty는 entropy 하한의 수학적 보장이 아닙니다. 예를 들어 h0 = 0.8은 N = 64에서 유효 위치 수 exp(h×log64)가 약 27.9보다 작아질 때 벌점을 주는 기준입니다. 이 숫자는 실제 실험 결과가 아니라 정의에서 계산한 해석입니다.

## 차별성과 선행연구 범위

Attention-based Deep MIL은 학습형 가중 집계를 제시했고, mixed/gated pooling 연구도 존재합니다. AEM은 병리 영상 MIL에서 entropy 최대화로 attention 과집중을 다룹니다. Transformer attention entropy collapse를 다루는 연구도 있습니다. 따라서 **entropy를 높이면 좋다는 아이디어 자체를 최초 기여로 쓰지 않습니다**.

이 프로젝트의 목표 기여는 작은 CNN 분류에서 다음을 함께 공개하는 것입니다.

- 동일 초기화·데이터 순서·증강으로 풀링 방법을 비교한 실험
- 가림 면적·채움·반복이 고정된 평가 프로토콜
- clean 정확도와 가림 정확도의 순위 및 절충
- 실제 공간 entropy와 penalty 활성 기록
- 학습 시드 단위 불확실성과 고정 혼합 대조군
- 감사 가능한 이미지별 예측과 자동 생성 코드

현재 검색만으로 논문 수준의 완전한 신규성 조사를 마쳤다고 보장할 수 없습니다. 투고 전 최근 global pooling, entropy regularization, occlusion robustness 연구를 다시 확인해야 합니다. 특히 thresholded entropy penalty의 신규성은 별도로 확인하십시오. 새 방법을 강하게 주장하려면 선행연구 대비 차이와 충분한 효과가 모두 필요합니다.

## 권장 실험 범위

| 단계 | 설정 | 횟수 | 역할 |
|---|---|---|---|
| 실행 점검 | 합성 이미지 2 epoch | 7 methods × 1 seed | 코드 확인만 수행 |
| 실제 pilot | CIFAR-100, ResNet-20, 10 epoch | 3 methods × 1 seed | loss와 학습 흐름 확인 |
| 본 실험 | CIFAR-100, ResNet-20, 80 epoch | 7 methods × 5 seeds = 35 | 주 결과와 paired 비교 |
| 데이터 확인 | CIFAR-10, ResNet-20, 80 epoch | 4 methods × 3 seeds = 12 | 권장 후속 검증 |
| backbone 확인 | CIFAR-100, ResNet-32, 80 epoch | 4 methods × 3 seeds = 12 | 시간 여유 시 추가 |
| 증강 대조군 | CIFAR-100, train square occlusion | 3 methods × 3 seeds = 9 | 시간 여유 시 추가 |
| 민감도 | train-derived validation, 40 epoch | 8 settings × 1 seed | 개발 분석과 선택 근거 |

본 실험 35회와 CIFAR-10 확인 12회가 권장 시작 범위입니다. 나머지 실험을 전부 필수로 실행하면 로컬 계산량이 커집니다. M5에서의 실행 시간은 동봉 benchmark로 확인하고 범위를 결정하십시오. 80 epoch에서 명확히 미수렴하면 공식 test 평가 전에 학습 길이를 다시 고정하고 모든 비교군에 동일하게 적용합니다.

기존 논문은 3개 시드·한 데이터셋·한 backbone입니다. 새 연구는 기본 5개 시드, 독립 validation, test 평가, 가림 강건성, calibration, 공간 진단, 다른 데이터셋 확인을 통해 증거의 범위를 넓힙니다. CIFAR로 바꿨다는 사실만으로 더 완성도 있는 논문이 되는 것은 아닙니다.

## 논문 표와 그림 구성

| 항목 | 내용 | 생성 파일 |
|---|---|---|
| 표 1 | dataset, split, optimizer, schedule, seeds | protocol.json과 split_summary.json |
| 표 2 | clean 및 평균 가림 정확도, SD, NLL, ECE | paper_table.csv |
| 표 3 | Floor와 3개 대조군의 paired 차이, CI, Holm p | paired_seed_tests.csv |
| 표 4 | 파라미터 수, 제한된 MAC 수, 장치별 시간 | complexity.csv와 latency_summary.csv |
| 그림 1 | 모델 구조와 entropy penalty | architecture_entropy_floor.pdf |
| 그림 2 | 학습과 clean validation 곡선 | learning_curves.pdf |
| 그림 3 | 실제 가림 면적별 정확도 | occlusion_curves.pdf |
| 그림 4 | 공간 entropy와 학습 중 하한 활성 비율 | spatial_entropy.pdf와 training_entropy_diagnostics.pdf |
| 그림 5 | 고정 ID 예시의 실제 pooling weight | pooling_weights_*.pdf |
| 보조 자료 | calibration, noise, confusion, per-class 분석 | 나머지 figures와 runs/test CSV |

코드는 정확한 설정에 맞는 모델 구조도, 데이터 차트, 실제 이미지 예시를 만듭니다. 최종 논문에서 방법의 표현이나 구성을 변경하면 구조도도 함께 확인하십시오. 클래스 가중치가 아닌 spatial weights를 Grad-CAM 또는 설명의 정답처럼 제시하지 않습니다.

## 논문 본문 순서

1. Introduction에서는 clean 성능만으로 pooling을 고르면 가림 상황에서의 정보 손실을 설명하기 어렵다는 질문을 제시합니다. 기존 DynaPool 연구를 정확히 인용하고 이번 연구가 공간 축을 조사한다고 구분합니다.
2. Related Work에서는 global pooling, mixed pooling, attention aggregation, entropy regularization, masking augmentation을 정리합니다. AEM과 다른 것은 이미지 해상도, 문제 설정, 통제 비교, thresholded penalty와 평가 조건이라고 한정합니다.
3. Method에서는 7개 풀링과 loss를 제시합니다. floor는 작은 탐색적 변형이고 novel SOTA 모듈로 포장하지 않습니다.
4. Experimental Setup에서는 split, mask 생성 규칙, 실제 면적, 최종 epoch 규칙, 시드, 통계 단위와 profiling 제외 항목을 명확히 씁니다.
5. Results에서는 주 지표부터 제시하고 clean 절충, penalty 활성, 고정 혼합 비교, 다른 데이터셋 결과를 차례로 봅니다.
6. Discussion에서는 상관과 인과를 구분하고, 실제 가림과 합성 가림 차이, 작고 유사한 CIFAR 데이터셋, 모델 범위, 적은 반복, MPS 비결정성을 설명합니다.
7. Conclusion은 측정된 관찰만 요약합니다. 결과에 없는 실세계 안전성이나 보편적 강건성 향상을 주장하지 않습니다.

## 결과에 따른 판단

- Floor가 우수하고 다른 데이터셋에서도 유지되면, 과도한 집중을 제한하는 간단한 제어와 그 절충을 중심으로 작성합니다. clean 손실도 함께 보고합니다.
- Entropy와 Floor가 비슷하면, 특정 하한 제어의 우월성보다 정규화와 공간 집중도의 관계를 중심으로 작성합니다.
- 고정 혼합이 충분하면, 복잡한 정규화가 필요한 조건과 불필요한 조건을 논의합니다.
- Floor penalty가 작동하지 않으면, 이 설정에서 공간 집중 문제가 관찰되지 않았다고 보고합니다. test를 본 뒤 기준을 계속 높여 효과를 만들어내지 않습니다.
- 모든 개선이 불확실하면, 새로운 방법의 우위 논문 대신 평가·진단 중심 실증 논문으로 범위를 한정하거나 추가 증거를 수집합니다. 통계적으로 불확실한 최고 평균을 성능 향상으로 단정하지 않습니다.

## 참고문헌과 공식 자료

- Lee, C.-Y., Gallagher, P. W., & Tu, Z. (2016). Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree. AISTATS. https://proceedings.mlr.press/v51/lee16a.html
- Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. ICML. https://proceedings.mlr.press/v80/ilse18a.html
- Zhang, Y., Li, H., Sun, Y., Shui, Z., Li, J., Zhu, C., & Yang, L. AEM: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification. arXiv:2406.15303, revised 2025, accepted MICCAI 2025. https://arxiv.org/abs/2406.15303
- Zhai, S., et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. ICML. https://proceedings.mlr.press/v202/zhai23a.html
- DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. https://arxiv.org/abs/1708.04552
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. https://arxiv.org/abs/1512.03385
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. 데이터와 원 보고서: https://www.cs.toronto.edu/~kriz/cifar.html
- GeM 원 연구: Radenović, F., Tolias, G., & Chum, O. Fine-tuning CNN Image Retrieval with No Human Annotation. IEEE TPAMI, 2019. https://doi.org/10.1109/TPAMI.2018.2846566
- PyTorch MPS 공식 문서: https://docs.pytorch.org/docs/stable/notes/mps.html

서지사항과 투고처의 스타일은 최종 원고에서 다시 확인하십시오. 기존 DynaPool의 최종 권호·DOI는 실제 발행 정보가 나온 뒤 추가합니다. 이 프로젝트는 어떤 투고처에도 자동으로 제출하거나 데이터를 업로드하지 않습니다.
