컴퓨터 비전 4주차 논문 요약 -- ImageNet Classification with Deep
Convolutional Neural Networks

2021312088 백인진

본 논문은 ImageNet 데이터셋 분류 문제에서 우승한 AlexNet 구조에 대해
설명하고 있다. AlexNet 설명에 앞서 사용한 데이터셋인 ImageNet을
설명한다. 이는 22000개의 범주를 가지며 약 1500만개의 고해상도 이미지를
포함한다. AlexNet 알고리즘이 우승한 대회는 ILSVRC로, ImageNet 데이터셋의
서브세트를 활용해, 1000개의 분류에 대해 각 1000개씩의 이미지를 활용하게
된다. 이로 인해, 약 120만개의 학습 데이터, 5만개의 검증 데이터, 15만개의
테스트 데이터를 갖는다. 해당 이미지에 대해 256\*256의 동일한 크기로
고정시키는데, 이미지의 가로, 세로 중 짧은 쪽을 256픽셀로 맞추고 중앙
부분을 256\*256 크기로 잘라냈다. 이후 이미지 각 픽셀값에 대해 학습
데이터의 평균 값을 빼 normalize하는 전처리 과정을 거쳤다.

다음으로 AlexNet 구조에 대해 설명한다. 간단히 설명하면, max pooling을
활용하는 convolutional layer 5개와 fully connected layer 3개로 구성된다.
이를 자세히 단계별로 설명하면 아래와 같다.

입력층에서 224\*224\*3 크기의 이미지를 받아 다음 레이어인 Convolution
레이어에서 11\*11 크기의 96개 필터를 사용하여 55\*55\*96의 출력을
만든다. 이후 max pooling 레이어에서 3\*3 크기의 96개 필터를 사용해
27\*27\*96의 출력을 만든다. 다음은 Local Response Normalization(LRN)을
활용한 normalization 레이어다. LRN은 일반화를 목적으로 하며, sigmoid
혹은 tanh 함수는 입력 데이터 간 속성의 편차가 심하면 활성화함수의 평평한
부분에 몰리게(saturating현상) 되어 vanishing gradient 문제를 유발할 수
있다. 이에 반해, ReLU함수는 non-saturating nonlinearity 함수이기 때문에
saturating을 방지하기 위한 정규화가 필요하지 않다. 그렇지만 양수값을
그대로 전달하는 특성이 있어 큰 값이 전달될 경우 주변의 낮은 값들이
뉴런에 전달되는 것을 방해할 수 있다. 이러한 상황을 예방하기 위한
정규화가 LRN이다. 즉, 강한 자극이 주변의 약한 자극을 전달하는 것을 막는
효과인 측면 억제를 방지하기 위한 정규화 과정이다. 다시 AlexNet의 구조를
이어 설명하면, LRN 정규화를 활용하여 27\*27\*96 입력을 정규화 해 동일
크기의 27\*27\*96 출력을 생성한다. 이후 Convolution layer를 통해 5\*5
크기의 필터 256개를 활용하여 27\*27\*256 출력을 만들고 max pooling
레이어를 지나며 3\*3크기 필터를 활용해 13\*13\*256 출력을 만든다. 이는
다시 LRN을 사용한 normalization layer를 지나 13\*13\*256 출력을 낸다.
이는 다음 convolutional layer에서 3\*3\*256 필터 384개를 이용하여
13\*13\*384의 출력을 만들고 한번 더 convolutional layer를 거쳐 3\*3\*192
필터 384개를 활용해 13\*!3\*384 feature map을 생성한다. 또 다음
convolutional layer에서 3\*3\*192 필터 256개를 거쳐 13\*13\*256 출력을
얻은 뒤 max pooling layer에서 3\*3 필터를 활용해 6\*6\*256 출력을
만든다. 다음으로 fully connected layer를 두번 거치며 6\*6\*256 입력에서
4096 출력을, 4096 입력에서 4096 출력을 생성한다. 마지막으로 output
layer에서 fully connected layer를 활용해 1000개의 출력을 구성한다.

이러한 구조의 AlexNet은 기존 구조와 다른 특징을 갖는다. 먼저 활성화
함수를 ReLU 함수를 활용했다는 점이다. 이는 tanh 함수에 비해 약 6배 정도
빠른 속도를 보이며, AlexNet 이후 가장 많이 활용되는 활성화함수가 된다.
다음으로 과적합을 막기 위해 dropout을 활용한다. 이는 fully connected
layer의 뉴런들 중 일부의 값을 0으로 변경하며 학습을 진행해, 해당
뉴런들이 feed forward, back propagation 과정에 아무런 영향을 미치지
못하게 하는 것이다. 이는 train 과정에서만 활용되고, test 과정에서는 모든
뉴런을 활용한다. Overlapping pooling 또한 활용한다. LeNet-5의 경우,
average pooling을 사용하였는데 AlexNet에서는 max pooling을 활용하였고
이러한 Pooling filter를 크기에 비해 stride를 짧게 주며 중첩되도록
사용하였다. 이럴 경우 Top-1, Top-5 에러 비율을 줄이는데에 효과가 있다.
또한 위에서 설명한 LRN을 사용했다는 특징을 갖는다. 더불어 과적합을
막기위해 데이터의 양을 늘리고자 했는데 이때 data augmentation 방식을
활용하였다. 이는 동일한 이미지를 좌우반전시키거나, 잘라내어 다른
이미지로 만들어내는 기법이다.

이러한 특징을 통해 기존의 알고리즘에 비해 Top-5 test error 값을 약 11%
향상시킨 성능을 보일 수 있게 된 것이다. 더불어 학습 과정에서 2개의 GPU를
사용해 모델을 병렬 학습 시키며 딥러닝에 있어 GPU 구현이 보편화되기도
했다. 더불어 과적합을 방지하기 위한 다양한 기법을 소개해 CNN기반의
알고리즘의 발전을 불러 일으켰다.
