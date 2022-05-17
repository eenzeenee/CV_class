VGG

input : 224\*224\*3 사이즈 고정

preprocessing : 각 픽셀에 대해 training set에서 계산된 mean RGB값 빼기

stride : 1로 고정

padding : 3\*3 conv layer에 대해 1 pixel로 고정

max-pooling : 2\*2 window에서 stride 2로 수행됨

conv layer stack 뒤에 3개의 fully-connected layer 위치 (4096 - 4096 -
1000)

모든 hidden layer : ReLU 활성화함수

1)  A![](media/image2.png){width="6.345582895888014in"
    > height="7.09729002624672in"}

-   input : 224\*224\*3 input image

-   conv 1 : 64개의 3\*3 필터 커널 + zero padding 1 + stride 1 (zero
    > padding, stride 설정 모두 동일)

-   pool 1 : stride 2로 max pooling ; 224\*224\*128 -\> 112\*112\*128

-   conv 2 : 128개의 3\*3 필터 커널 사용

-   pool 2 : stride 2로 max pooling : 112\*112\*128 -\> 56\*56\*128

-   conv 3 : 256개의 3\*3 필터 커널 사용

-   conv 4 : 256개의 3\*3 필터 커널 사용

-   pool 3 : stride 2로 max pooling : 56\*56\*128 -\> 28\*28\*128

-   conv 6: 512개의 3\*3 필터 커널 사용

-   conv 7 : 512개의 3\*3 필터 커널 사용

-   pool 4 : stride 2로 max pooling : 28\*28\*128 -\> 14\*14\*128

-   conv 8: 512개의 3\*3 필터 커널 사용

-   conv 9 : 512개의 3\*3 필터 커널 사용

-   pool 5 : stride 2로 max pooling : 28\*28\*128 -\> 7\*7\*128

-   fully-connected 1 : 7\*7\*128 특성맵을 flatten하여 4096개의 뉴런과
    > fully connected (dropout 적용됨)

-   fully-connected 2 : fully-connected 1의 4096개의 뉴런과 4096개의
    > 뉴런이 fully connected (dropout 적용됨)

-   softmax : 1000개의 뉴런으로 구성되며 fully-connected 2와 fully
    > connected됨. 출력값들은 softmax함수로 활성화됨.

ResNet

갑자기 어느순간 error가 큰 폭으로 줄어드는 것을 확인
가능![](media/image3.png){width="6.267716535433071in"
height="2.111111111111111in"}

*Is learning better networks as easy as stacking more layers?*

vanishing 문제를 해결하기 위해 x 를 다시 input하여
보간![](media/image1.png){width="2.8333333333333335in"
height="1.5104166666666667in"}

residual learning : 기존의 weighted sum + short cut connections to match
the dimensions

![](media/image4.png){width="4.707843394575678in"
height="10.01562554680665in"}
