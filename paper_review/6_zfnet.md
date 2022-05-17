ZF Net : 기본적으로 AlexNet과 유사한 구조

-   deconvnet을 추가하여 각각의 stage에서 어떠한 상황인지 시각화하고자
    > 함

![](media/image1.png){width="4.859375546806649in"
height="4.592997594050743in"}

-   **Unpooling**

> 기존의 Max pooling 연산 : 비가역적 - 가장 강한 자극만을 다음단계로
> 전달하여 다음 단계에서 어느 신호가 가장 강한 자극을 가지고 있었는지
> 파악 불가능
>
> 이를 파악하기 위해 switch 개념 도입
>
> switch : Maxpooling 전에 가장 강한 자극의 위치 정보를 저장하는 것
>
> unpooling을 진행하며 switch 정보를 활용하여 가장 강한 자극의 위치로
> 돌아갈 수 있도록 함

-   **Rectification**

> 활성화 함수 ReLU는 feature map이 활성화 함수를 거치고 나면 항상
> 양수값을 가지도록 보장
>
> 이때, 음수 정보에 대해 재건할 수 없음. 그러나 본 논문에서는 음수부분이
> 우리가 원하는 자극을 찾는 데에 크게 영향을 미치지 않으므로 크게 문제
> 없다고 밝힘

-   **Filtering**

> Convolutional Network에서 이전 레이어로부터 feature map을 합성곱하기
> 위해 학습한 필터를 사용함. 이를 역으로 연산하기 위해서는 filter들이
> transposed된 내용을 사용함. 수학적으로 가역 연산이 가능하므로 문제
> 없음
>
> AlexNet 구조
>
> ![](media/image3.png){width="4.119792213473316in"
> height="3.2042825896762905in"}
>
> zfnet 구조
>
> ![](media/image2.png){width="6.267716535433071in"
> height="1.5138888888888888in"}
>
> -\> AlexNet 구조와의 차이점

1.  첫번째 Convolution layer가 11\*11 크기의 필터를 stride 4 였던 내용을
    > 7\*7 크기의 필터를 stride 2로 변경

2.  3, 4, 5 번째 Convolution layer의 depth를 384, 384, 256에서 512,
    > 1024, 512로 변경함.

3.  AlexNet에서는 병렬 GPU를 사용하여 3, 4, 5 번째 Convolution layer에서
    > 희소 연결이 나타나는 반면, zfnet에서는 고밀도의 연결로 변경됨

4.  

-\> 필터의 크기는 줄이면서, 개수를 늘리는 방향으로 hyper parameter를
조정하여 AlexNet을 개선

-\> 더불어 Deconvolution layer를 추가하여 ConvNet의 활동을 시각화함

LeNet -\> AlexNet 으로 넘어가며 어떤 변화를 진행하며 성능을 좋게
만들었는지

-   레이어를 더 깊게 쌓아서

-   활성화함수의 변경 : 시그모이드 -\> ReLU : 계산 더 편리해짐

-   정규화 추가 + pooling 변화 : average -\> max : 계산 더 편리해짐

-   커널의 크기 변화

-   dropout 추가 : 과적합 방지를 위해서

    -   과적합은 왜 일어나지? -\> train data가 파라미터에 너무 많이
        > 반영되는 경우 -\> 필요 이상의 파라미터가 있어서 이를 제거하기
        > 위해 dropout 진행
