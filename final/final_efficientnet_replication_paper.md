[EfficientNet: Rethinking Model Scaling for Convolutional Neural
Networks](https://arxiv.org/pdf/1905.11946.pdf)

-   Replication paper

2021312088 백인진

-   introduction

> Convolution layer의 성능을 올리기 위해 scaling up을 시도하는 것은 매우
> 일반적인 일이다. 예를 들어, ResNet의 경우 ResNet-18부터 ResNet-200까지
> 레이어의 깊이를 늘려 성능의 향상을 이루어냈다. 최근, G-Pipe를 활용한
> 모델의 경우 ImageNet top-1 accuracy를 84.3% 까지 향상시켰다. 이를 통해
> scaling up이 성능 향상에 중요한 요소임을 알게 되었으나, scaling up에
> 대한 과정의 이해도는 매우 낮은 수준이다. scaling up 방식에는 크게
> 3가지가 있다. 먼저 레이어의 깊이를 늘리는 것, channel의 넓이를 늘리는
> 것, 마지막으로 입력 이미지의 resolution을 높이는 것이다. 지난
> 연구에서는, 깊이, 넒이, 이미지 사이즈 이 세 가지 차원에 대해 scale하는
> 것이 일반적이었다. 임의적으로 차원 2, 3개를 scale할 수 있지만,
> 임의적인 scaling은 지루한 수동 튜닝이 필요하며 여전히 최적의 정확도와
> 효율성을 제공하지 못하는 경우가 많다.
>
> 이 논문에선, 위에서 이야기한 깊이, 넓이, 이미지 사이즈의 scaling up에
> 대해 최적의 조합을 찾고자 한다. 놀랍게도, 이와 같은 균형은 각각을
> 일정한 비율로 단순히 확장함으로써 달성할 수 있다. 이러한 관찰을
> 기반으로 단순하면서도 효과적인 compound scaling method를 제안한다.
> 이러한 요인을 임의로 스케일링하는 기존의 관행과 달리, 고정된 스케일링
> 계수 세트를 사용하여 네트워크 폭, 깊이 및 해상도를 균일하게
> 스케일링한다. 예를 들어, 만약 2\^N배 더 많은 계산 자원을 사용하고
> 싶다면, 네트워크 깊이를 α\^N만큼, 너비를 β\^N만큼, 이미지 크기를
> γ\^N만큼 증가시키면 된다. 여기서 α,β,γ는 원래 작은 모델의 작은 그리드
> 검색에 의해 결정되는 상수 계수이다.
>
> 직관적으로, compound scaling method은 의미가 있다. 입력 이미지가 클
> 경우 네트워크는 수용 필드를 증가시키기 위해 더 많은 레이어가 필요하고
> 더 큰 이미지에 더 미세한 패턴을 캡처하기 위해 더 많은 채널이 필요하기
> 때문이다. 기존의 이론적, 경험적 결과를 보아도, 모두 네트워크의 깊이와
> 넓이 사이의 특정한 관계가 있음을 보여준다. 그러나, 논문의 저자가 아는
> 한, 네트워크의 깊이와 넓이, 입력 이미지의 해상도에 대해 3가지 특성의
> 관계를 경험적으로 정량화한 것은 처음이다.
>
> 이 논문은 compound scaling method 방법을 MobileNets, ResNet에서 잘
> 작동하는 것을 보인다. 특히, 모델 확장의 효과는 베이스라인 네트워크에
> 크게 좌우된다. 더 나아가 neural architecture search를 사용하여 새로운
> 베이스라인 네트워크를 개발하고 EfficientNets 모델을 얻기 위해
> 확장한다. 특히 EfficientNet-B7은 현존하는 최고의 GPIpe 정확도(Huang
> 등, 2018)를 능가하지만 매개 변수를 8.4배 적게 사용하고 추론에서 6.1배
> 빠르게 실행한다. 널리 사용되는 ResNet-50(He 등, 2016)과 비교하여
> EfficientNet-B4는 유사한 FLOPS로 상위 1위 정확도를 76.3%에서
> 83.0%(+6.7%)로 향상시킨다. ImageNet 외에도 EfficientNets는 널리
> 사용되는 데이터 세트 8개 중 5개에서 전이가 잘되고 state-of-the-art
> 정확도를 달성하면서 기존 ConvNet보다 매개 변수를 최대 21배까지 줄인다.

-   Compound Model Scaling

> 이 절에서는 scaling 문제를 공식화하고 다양한 접근 방식을 연구하며
> 새로운 scaling 방법을 제안한다.

1)  Problem Formulation

> ConvNet Layer i는 함수로 정의할 수 있다. Yi = Fi(Xi), 여기서 Fi는
> 연산자이고, Yi는 출력 텐서이며, Xi는 입력 텐서이며, 텐서 모양은 \<Hi,
> Wi, Ci\>이며, 여기서 Hi와 Wi는 공간 차원이고 Ci는 채널 차원이다.
> ConvNet N은 composed layer의 목록으로 나타낼 수 있습니다. N = Fk
> ⊙\...⊙F2 ⊙F1(X1) = Δj=1\...k Fj(X1). 실제로, ConvNet 계층은 종종 여러
> 단계로 분할되고 각 단계의 모든 계층은 동일한 아키텍처를 공유한다. 예를
> 들어, ResNet에는 5개의 단계가 있으며, 각 단계의 모든 레이어는 첫 번째
> 레이어가 다운샘플링을 수행하는 것을 제외하고 동일한 컨볼루션 유형을
> 가집니다. 그러므로 ConvNet을 으로 정의할 수 있다. 여기서 FLI는 층 Fi가
> 1단계에서 Li를 반복하는 것을 나타낸다. \<Hi, Wi, Ci\> 는 레이어 i의
> 입력 텐서 X의 형상을
> 나타낸다.![](media/image1.png){width="1.0729166666666667in"
> height="0.20833333333333334in"}
>
> 대부분 최상의 계층 아키텍처 Fi를 찾는 데 초점을 맞춘 일반적인 ConvNet
> 설계와 달리 모델 스케일링은 기본 네트워크에 사전 정의된 Fi를 변경하지
> 않고 네트워크 길이(Li), 폭(Ci) 및/또는 해상도(Hi,Wi)를 확장하려고
> 한다. Fi를 수정함으로써 모델 스케일링은 새로운 자원 제약에 대한 설계
> 문제를 단순화하지만, 여전히 각 레이어에 대해 서로 다른 Li, Ci, Hi,
> Wi를 탐색할 수 있는 큰 설계 공간으로 남아 있다. design space를 더욱
> 줄이기 위해 모든 레이어가 일정한 비율로 균일하게 확장되어야 한다고
> 제한한다. 이 논문의 목표는 최적화 문제로 공식화될 수 있는 주어진 자원
> 제약에 대한 모델 정확도를 최대화하는 것이다.
> ![](media/image2.png){width="2.1097462817147856in"
> height="0.7032491251093613in"}
>
> 여기서 w,d,r은 네트워크 폭, 깊이 및 분해능 스케일링을 위한 계수이다.
> f\^i, l\^i, h\^i, w\^i, c\^i는 기준 네트워크에서 사전 정의된
> 파라미터이다.

2)  Scaling Dimension

> 문제 2의 주요 어려움은 최적의 d, w, r이 서로 의존하며 다른 자원 제약에
> 따라 값이 변한다는 것이다. 이러한 어려움 때문에 기존 방법은 대부분
> 다음 차원 중 하나로 ConvNets를 확장한다.
>
> ***Depth(d)*** : 네트워크 깊이 scaling은 많은 ConvNet에서 가장
> 일반적으로 사용되는 방법이다. 직관적인 이유는 ConvNet이 더 풍부하고
> 복잡한 기능을 포착하고 새로운 작업을 잘 일반화할 수 있다는 것이다.
> 그러나, 더 깊은 네트워크는 vanishing gradient problem로 인해
> 훈련하기가 더 어렵다. skip connections and batch normalization와 같은
> 몇 가지 기술이 훈련 문제를 완화하지만 매우 심층적인 네트워크의 정확도
> 향상은 감소한다. 예를 들어, ResNet-1000은 훨씬 더 많은 레이어를 가지고
> 있음에도 불구하고 ResNet-101과 비슷한 정확도를 가지고 있다.
>
> ***Width(w)*** : 네트워크 폭의 scaling은 일반적으로 작은 크기의 모델에
> 사용된다. (Zagoruyko & Komodakis, 2016)에서 논의된 바와 같이, 더 넓은
> 네트워크는 더 미세한 특징을 포착할 수 있고 훈련하기 쉬운 경향이 있다.
> 그러나 매우 넓지만 얕은 네트워크는 더 높은 수준의 특징을 포착하는 데
> 어려움을 겪는 경향이 있다. 우리의 경험적 결과는 네트워크가 w가 클수록
> 훨씬 넓어지면 정확도가 빠르게 saturation된다는 것을 보여준다.
> \[saturation = vanishing gradient problem\]
>
> ***Resolution(r)*** : 더 높은 해상도의 입력 이미지를 통해 ConvNets는
> 잠재적으로 더 미세한 패턴을 캡처할 수 있다. 초기 ConvNets의
> 224x224에서 시작하여, 현대의 ConvNets는 더 나은 정확도를 위해
> 299x299(Szegedy 등, 2016) 또는 331x331(Zoph 등, 2018)을 사용하는
> 경향이 있다. 최근, GPpie의 경우 480\*480 해상도의 입력 이미지를
> 활용하여 ImageNet에 대해 state-of-the-art 수준의 정확도를 보였다. 보다
> 높은 해상도, 600\*600의 경우 object detection을 위한 ConvNet에 널리
> 활용된다. 그러나, 매우 높은 해상도에 대해서는 정확도가 감소한다.
>
> 위의 분석들은 첫번째 관찰 결과를 나타낸다.
>
> ***Observation 1*** : 네트워크 폭, 깊이 또는 해상도의 크기를 늘리면
> 정확도는 향상되지만 대형 모델의 경우 정확도가 저하된다.

3)  Compound Scaling

> 이 논문에서 저자들은 다른 scaling dimension이 독립적이지 않다는 것을
> 경험적으로 관찰한다. 직관적으로 고해상도 이미지의 경우 네트워크 깊이를
> 키워서 receptive fields가 클수록 더 큰 이미지에 더 많은 픽셀을
> 포함하는 유사한 기능을 캡처할 수 있도록 해야 한다. 이에 따라 고해상도
> 이미지에 더 많은 픽셀로 더 미세한 패턴을 캡처하기 위해 해상도가
> 높을수록 네트워크 폭도 증가해야 한다. 이러한 직관은 우리가 기존의
> 1차원에 대한 스케일링이 아닌 다른 스케일링 차원을 조정하고 균형을 맞출
> 필요가 있음을 시사한다. 직관을 검증하기 위해 다양한 네트워크 깊이와
> 해상도에서 폭 배율을 비교하였다. 깊이(d)와 해상도(r)를 변경하지 않고
> 네트워크 폭(w)만 확장하면 정확도가 빠르게 saturation된다. 더 깊은
> 레이어 및 더 높은 해상도의 경우, 동일한 FLOPS\[초당 부동소수점 연산 :
> 컴퓨터가 1초동안 수행할 수 있는 부동소수점 연산의 횟수를 기준으로 성능
> 수치\] 비용으로 폭 배율이 훨씬 더 높은 정확도를 달성합니다. 이러한
> 결과는 두 번째 관찰로 이어진다.
>
> ***Observation 2*** : 더 나은 정확성과 효율성을 추구하기 위해서는
> ConvNet 스케일링 중에 네트워크 폭, 깊이 및 해상도의 모든 차원을 균형
> 있게 조정하는 것이 중요하다.
>
> 실제로 몇 가지 이전 작업들은 이미 네트워크 폭과 깊이의 균형을 임의로
> 조정하려고 시도했지만 모두 수동 조정이 필요하다. 이 논문은 네트워크
> 폭, 깊이 및 해상도를 원칙적으로 균일하게 스케일링하기 위해 복합 계수
> φ를 사용하는 새로운 Compound Scaling 방법을
> 제안한다.![](media/image3.png){width="2.1145833333333335in"
> height="1.1982633420822397in"}
>
> 여기서 α, β, γ는 작은 grid search로 결정될 수 있는 상수이다.
> 직관적으로, φ는 모델 스케일링에 사용할 수 있는 리소스 수를 제어하는
> 사용자 지정 계수이며, α, β, γ는 네트워크 폭, 깊이 및 해상도에 각각
> 이러한 추가 리소스를 할당하는 방법을 지정한다. 일반적인 ConvNet 실행의
> FLOPS는 네트워크의 깊이(d)를 두 배로 늘리면 FLOPS가 두 배로
> 증가하지만, 네트워크의 폭(w), 이미지의 해상도(r)를 두 배로 늘리면
> FLOPS가 네 배로 증가한다. 일반적으로 ConvNets에서 컨볼루션 작업이 계산
> 비용을 지배하기 때문에 방정식 3으로 ConvNet을 확장하면 총 FLOPS가 대략
> (α · β\^2 · γ\^2)φ만큼 증가한다. 이 논문에서는 α · β\^2 · γ\^2의 값을
> 2와 유사하게 제한하여 새로운 φ에 대하여 FLOPS가 약 2\^φ배 증가하도록
> 한다.

-   EfficientNet Architecture

> 모델 스케일링은 기준선 네트워크에서 layer 연산자 Fˆi를 변경하지 않기
> 때문에 양호한 baseline network를 갖는 것 또한 중요하다. 기존
> ConvNets를 사용하여 스케일링 방법을 평가할 것이지만 스케일링 방법의
> 효과를 더 잘 입증하기 위해 EfficientNet이라는 새로운 mobile-size
> baseline도 개발했다.
>
> (Tan et al., 2019)에서 영감을 받아 정확성과 FLOPS를 모두 최적화하는
> 다목적 신경 아키텍처 검색을 활용하여 baseline network를 개발한다.
> 특히, 우리는 (Tan et al., 2019)와 동일한 검색 공간을 사용하고 최적화
> 목표로 ACC(m)×\[FLOPS(m)/T\]\^w를 사용한다. 여기서 ACC(m)와 FLOPS(m)는
> 모델의 정확도를 나타내고 T는 대상 FLOPS이고 w=0.07은 하이퍼 파라미터
> 제어이다. (Tan et al, 2019; Cai et al, 2019)와는 달리, 여기서는 특정
> 하드웨어 디바이스를 목표로 하지 않기 때문에 지연 시간보다는 FLOPS를
> 최적화한다. 이 연구에서는 EfficientNet-B0이라는 이름의 효율적인
> 네트워크를 생성한다. (Tan et al, 2019)와 동일한 검색 공간을 사용하기
> 때문에 EfficientNet-B0이 더 큰 FLOPS 대상 때문에 약간 더 크다는 점을
> 제외하면 아키텍처는 MnasNet과 유사하다. baseline EfficientNet-B0부터
> 시작하여 복합 스케일링 방법을 적용하여 다음의 두 단계로 스케일업한다.

-   Step 1 : 먼저 사용 가능한 자원이 두 배 더 많다고 가정하고 γ = 1을
    > 고정하고 위에서 세운 방정식을 기반으로 α, β, γ의 small grid
    > search를 수행한다. 특히 EfficientNet-B0에 대한 최상의 값은 α·β2·γ2
    > ≈2.의 제약 조건 하에서 α = 1.2, β = 1.1, γ=1.15이다.

-   Step 2 : 그런 다음 α, β, γ를 상수로 고정하고 위의 두번째 방정식을
    > 사용하여 다른 θ로 기준선 네트워크를 확장하여 EfficientNet-B1에서
    > B7까지 구한다.

> 특히 대형 모델을 중심으로 직접 α, β, γ를 검색하면 훨씬 더 나은 성능을
> 얻을 수 있지만 대형 모델일수록 검색 비용이 매우 비싸다. 이 방법은
> 소규모 기준선 네트워크에서 검색을 한 번만 수행한 다음(Step 1) 다른
> 모든 모델(Step 2)에 대해 동일한 스케일링 계수를 사용하여 이 문제를
> 해결한다.

-   Experiments

> 이 섹션에서는 먼저 기존 ConvNets와 새로 제안된 EfficientNets에 대한
> 확장 방법을 평가할 것이다.

1)  Scaling Up MobileNets and ResNets

> 개념을 증명하기 위해, 먼저 널리 사용되는 MobileNets(Howard et al,
> 2017; Sandler et al, 2018)와 ResNet(He et al, 2016)에 스케일링 방법을
> 적용한다. 다른 1차원 스케일링 방법에 비해, 이 논문의 compound scaling
> 방법은 이러한 모든 모델의 정확도를 향상시켜 일반적인 기존 ConvNets에
> 대해 제안된 스케일링 방법의 효과를 시사한다.

2)  ImageNet Results for EfficientNet

> 이 논문에서는 (Tan et al. 2019)와 유사한 설정을 사용하여 ImageNet에서
> EfficientNet 모델을 훈련한다. 즉, RMSProp optimizer with decay 0.9 와
> momentum 0.9, batch norm momentum 0.99, weight decay 1e-5, initial
> learning rate 0.256 that decays by 0.97 every 2.4 epochs 의 설정을
> 사용하는 것이다. 또한, SiLU (Swish-1) activation, AutoAugment,
> stochastic depth with survival probability 0.8 을 사용한다. 큰
> 모델에는 더 많은 정규화가 필요하다고 일반적으로 알려진 바와 같이,
> EfficientNet-B0의 경우 0.2에서 B7의 경우 0.5로 dropout (Srivastava et
> al, 2014) 비율을 선형적으로 증가시킨다. training 세트에서 무작위로
> 선택한 25K개의 이미지를 minival 세트로 예약하고 이 minival에 대해 조기
> 중지를 수행한 다음 원래 validation 세트에서 조기 중지된 체크포인트를
> 평가하여 최종 validation 정확도를 보고한다.
>
> EfficientNet 모델은 일반적으로 유사한 정확도를 가진 다른 ConvNet보다
> 훨씬 적은 수의 매개 변수와 FLOPS를 사용한다. 특히 EfficientNet-B7은
> 66M 매개 변수와 37B FLOPS로 84.3%의 상위 1위 정확도를 달성하며, 이전
> 최고의 GPIpe보다 정확도는 높지만 8.4배 더 작다(Huang et al, 2018).
> 이러한 장점은 EfficientNet에 맞게 맞춤화된 더 나은 아키텍처, 더 나은
> 확장 및 더 나은 학습 설정을 모두 통해 얻을 수 있다.
>
> 지연 시간을 검증하기 위해 실제 CPU에서 몇 가지 대표적인 CovNets에 대한
> 추론 지연 시간을 측정했으며, 여기서 20회 이상의 평균 지연 시간을
> 보고하였다. EfficientNet-B1은 널리 사용되는 ResNet-152보다 5.7배
> 빠르게 실행되는 반면 EfficientNet-B7은 GPIpe(Huang et al, 2018)보다 약
> 6.1배 빠르게 실행되므로 EfficientNets는 실제 하드웨어에서 실제로
> 빠릅니다.

3)  Transfer Learning Results for EfficientNet

> 일반적으로 사용되는 전이 학습 데이터셋 목록에 대해 EfficientNet을
> 평가했다. 새로운 데이터셋에 ImageNet pretrained 체크포인트와
> finetuning을 취하는 (Kornblith et al., 2019) 와 (Huang et al., 2018)
> 등과 같은 train 설정을 차용한다. 전이학습에 대한 performance는 다음과
> 같다. (1) NASNet-A(Zoph et al, 2018) 및 Inception-v4(Szegedy et al,
> 2017)와 같은 공개 가용 모델과 비교하여, EfficientNet 모델은 평균
> 4.7배(최대 21배) 매개 변수 감소로 더 나은 정확도를 달성한다. (2) 훈련
> 데이터를 동적으로 합성하는 DAT(Ngiam et al, 2018)와 전문 파이프라인
> 평행화로 훈련된 GPIpe(Huang et al, 2018)를 포함한 최첨단 모델과
> 비교하여 EfficientNet 모델은 여전히 8개 데이터 세트 중 5개에서
> 정확도를 능가하지만 9.6배 적은 매개 변수를 사용한다. 일반적으로
> EfficientNets는 ResNet(Huang et al, 2016), DenseNet(Huang et al,
> 2017), Inception(Szegedy et al, 2017) 및 NASNet(Zoph et al, 2018)을
> 포함한 기존 모델보다 훨씬 적은 수의 매개 변수로 지속적으로 더 나은
> 정확도를 달성한다.

-   Discussion

> EfficientNet 아키텍처에서 제안된 스케일링 방법의 기여를 분리하여
> 확인하기 위해, 동일한 EfficientNet-B0 기본 네트워크에 대한 다양한
> 스케일링 방법의 ImageNet 성능을 비교한다. 일반적으로 모든 스케일링
> 방법은 더 많은 FLOPS의 비용으로 정확도를 향상시키지만, 우리의 복합
> 스케일링 방법은 다른 단일 차원 스케일링 방법보다 정확도를 최대
> 2.5%까지 더 향상시킬 수 있어 제안된 복합 스케일링의 중요성을 시사한다.
>
> 다음으로 compound scaling method가 다른 모델에 비해 나은 이유를 더 잘
> 이해하기 위해 스케일링 방법이 다른 몇 가지 대표적인 모델에 대한 클래스
> 활성화 맵 (Zhou et al, 2016) 을 비교한다. 이때 모든 모델은 동일한
> baseline에서 scaling된다. 이미지는 ImageNet validation 세트에서
> 무작위로 선택된다. compound scaling 방식을 사용하는 모델은 더 많은
> object detail 정보가 있는 관련 영역에 초점을 맞추는 경향이 있는 반면,
> 다른 모델은 object detail 정보가 부족하거나 이미지의 모든 object를
> 잡아낼 수 없는 것을 알 수 있다.

-   Conclusion

> 이 논문에서는 ConvNet scaling을 체계적으로 연구하고 네트워크의 폭,
> 깊이, 해상도에 대한 균형 조정이 중요하지만 기존 연구에서는 이에 대한
> 연구가 누락되어 정확성과 효율성을 향상시킬 수 없다는 단점을 지적한다.
> 이러한 문제를 해결하기 위해 모델의 효율성을 유지하며 baseline
> ConvNet을 보다 원칙적인 방식으로 target resource 제약으로 쉽게 확장할
> 수 있는 간단하면서도 매우 효과적인 compound scaling 방식을 제안한다.
> 이러한 compound scaling 방법에 의해, mobile-size인 EfficientNet 모델이
> ImageNet과 더불어 일반적으로 사용되는 5개의 전이학습 데이터에 대해
> 모두 훨씬 적은 수의 매개변수와 FLOPS로 state-of-the-art 정확도를
> 능가하는 매우 효과적인 scale-up이 될 수 있음을 보여준다.
