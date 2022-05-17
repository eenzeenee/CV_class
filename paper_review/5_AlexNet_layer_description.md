![](media/image1.png){width="6.267716535433071in" height="3.125in"}

레이어 1-1 : convolution layer

> \- 96개의 11\*11\*3 크기의 필터 커널 활용하여 입력 이미지 합성곱 진행
>
> \- Stride = 4, zero-padding 사용하지 않음
>
> \- Output : 55\*55\*96 크기의 feature map
>
> \- 활성화함수 : ReLU

-   ReLU를 사용하며, 예측 및 학습 속도가 빠르고 정확도가 유지되었음

-   non-saturating한 모양으로 sigmoid, tanh 함수의 gradient vanishing
    > 문제를 해결가능

레이어 1-2 : pooling layer

> \- 96개의 3\*3 크기의 overlapping max pooling 진행

-   overlapping pooling : 버리는 정보를 최소화하고 정보 추출을 보다 많이
    > 챙겨가려고

-   단순하게 선을 따고 코너를 찾는 것이 아니라 그 외의 다양한 형태에
    > 대한 정보를 추출해야 하므로 보다 복잡한 정보가 필요하여
    > overlapping pooling 사용

> \- Stride = 2
>
> \- Output : 27\*27\*96 크기의 feature map

레이어 1-3 : Local Response Normalization

> \- 일반화를 목적으로 하는 정규화
>
> \- ReLU 활성화함수를 활용하는 과정에서 양수값이 그대로 전달될 때 큰
> 값이 전달되는 경우 주변의 작은 값들이 뉴런에 전달되는 것을 강화하여 그
> 값을 왜곡할 수 있다. 이러한 상황을 예방하기 위한 정규화 과정
>
> \- 강한 자극으로 인해 주변의 약한 자극을 크게 전달하는 것을 막는
> 효과인 측면 억제를 방지하기 위한 정규화 과정
>
> \- Output : feature map의 변화 없음

레이어 2-1 : Convolution layer

> \- 256개의 5\*5\*48 크기 필터 커널을 활용하여 입력 feature map에 대해
> 합성곱 진행
>
> \- Stride = 1, zero-padding = 2
>
> \- Output : 27\*27\*256 크기의 feature map
>
> \- 활성화함수 : ReLU

레이어 2-2 : pooling layer

> \- 256개의 3\*3 크기의 overlapping max pooling 진행
>
> \- Stride = 2
>
> \- Output : 13\*13\*256 크기의 feature map

레이어 2-3 : Local Response Normalization

레이어 3. Convolution layer

> \- 384개의 3\*3\*256 크기 필터 커널을 활용하여 입력 feature map에 대해
> 합성곱 진행
>
> \- Stride = 1, zero-padding =1
>
> \- Output : 13\*13\*384 크기의 feature map
>
> \- 활성화함수 : ReLU

-   convolution layer에서 multiple GPU를 사용하여 나눠서 출력한 feature
    > map 간의 소통이 일어나는 곳 (유일)

-   GPU 크기의 한계로 kernel의 파라미터를 두 개의 GPU로 나누어 학습하여
    > 시간대비 우수한 결과 얻음

레이어 4 : Convolution layer

> \- 384개의 3\*3\*192 크기 필터 커널을 활용하여 입력 feature map에 대해
> 합성곱 진행
>
> \- Stride = 1, zero-padding =1
>
> \- Output : 13\*13\*384 크기의 feature map
>
> \- 활성화함수 : ReLU

레이어 5-1 : Convolution layer

> \- 384개의 3\*3\*192 크기 필터 커널을 활용하여 입력 feature map에 대해
> 합성곱 진행
>
> \- Stride = 1, zero-padding =1
>
> \- Output : 13\*13\*384 크기의 feature map
>
> \- 활성화함수 : ReLU

레이어 5-2 : pooling layer

> \- 256개의 3\*3 크기의 overlapping max pooling 진행
>
> \- Stride = 2
>
> \- Output : 6\*6\*256 크기의 feature map
>
> \-

레이어 6 : Fully connected layer

> \- 입력된 6\*6\*256 feature map을 flatten하여 9216차원의 벡터를 만듦
>
> \- 4096개의 뉴런과 fully connected한 뒤 그 결과를 ReLU함수로 활성화

레이어 7 : Fully connected layer

> \- 4096개의 뉴런으로 구성됨
>
> \- 레이어 6의 4096개 뉴런과 fully connected한 뒤 그 결과를 ReLU함수로
> 활성화

레이어 8 : Fully connected layer

> \- 1000개의 뉴런으로 구성됨
>
> \- 레이어 7의 4096개의 뉴런과 fully connected한 뒤 그 결과를 softmax에
> 적용하여 1000개 클래스 각각에 대한 확률을 구함

총 약 6천만개의 파라미터가 훈련되어야 함.

-   Multi GPU :

    -   두개의 GPU 나뉘어서 학습에 사용됨 - 각각 커널을 절반씩 나누어
        > 학습

    -   해당 과정을 분해해본 결과 색상 정보와 형태 정보를 나누어
        > 학습하는 것을 확인

LeNet과의 차이점

-   Multi GPU 사용

-   layer 개수의 차이

-   파라미터의 수

-   출력 layer의 차이 (rbf 사용 여부)

-   입력 이미지의 채널 개수

-   
