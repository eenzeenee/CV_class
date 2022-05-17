컴퓨터 비전 4주차 논문 요약 -- Gradient-Based Learning Applied to
Document Recognition

2021312088 백인진

본 논문은 CNN의 발전 계기가 된 LeNet 구조에 대해 설명하는 논문이다.
손글씨와 같은 고차원의 패턴을 가진 데이터를 분류해내는 문제에서 기존
패턴 인식 방법과 LeNet방식이 갖는 차이점에 초점을 맞추어 LeNet
알고리즘을 설명한다. 기존의 인식 방법은 수작업으로 특징을 직접 추출해낸
정보를 가지고 패턴과 연관된 정보와 그렇지 않은 정보를 분리하였다. 이를
사용하여 fully-connected network에 입력하였는데, 이때 입력 이미지의
크기가 커서 neural network의 가중치 파라미터 개수가 커지며 메모리 사용량
또한 증가하였다. 또한 입력 이미지의 이동, 회전, 왜곡 등의 모든 경우의
수를 고려하기 어려운 방식이다. 이러한 이유로 기존의 패턴 인식 방식은
많은 양의 학습 데이터를 필요로 한다. 이러한 단점에 대해 CNN은 모서리,
코너와 같은 특정한 local feature를 추출하여 입력 데이터의 다양성을
포용할 수 있다. CNN은 3가지 부분으로 구조가 형성된다. 먼저 local
receptive fields를 활용한다. 이는 고양이에게 어떤 그림을 보여줬을 경우,
특정 그림에서만 뉴런이 반응하는 것으로부터 아이디어를 얻은 내용으로,
입력 이미지의 local feature를 추출할 수 있도록 한다. 이를 위해 kernel
혹은 filter을 활용하는데, 이러한 과정으로 convolution한다고 하며
filter가 하나의 가중치로 작용한다. 이 가중치 값을 학습 과정에서 계속
업데이트하게 된다. Convolution 과정을 마치면 feature map을 얻게되고, 이
것을 다음 레이어에 보내며, fully connected가 아닌 local feature에 집중한
모델을 만들 수 있게 된다. 이 과정에서 기존 fully connected 모델에 비해
적은 수의 가중치를 갖게 된다. 다음으로 shared weight 개념 또한 활용한다.
Convolution을 위해 filter, kernel을 입력 데이터에 적용할 때, filter가
적용된 결과는 계속 변경되지만, filter는 변하지 않는 것을 의미한다.
앞에서 filter가 가중치로 작용한다고 했으므로, 가중치가 동일하게 공유되는
것을 알 수 있다. 마지막으로 subsampling방식을 활용한다. 이는 추출한
local feature에 대해서 입력 데이터의 변형에 관계 없는 global한 feature를
추출하기 위해 사용된다. 이 과정을 거친 뒤 classification을 진행하게
된다.

LeNet-5의 구조는 입력 레이어를 제외하고 총 7개의 레이어로 구성된다. 이전
LeNet-1에서는 28\*28 크기의 이미지를 입력으로 받았는데, LeNet-5에서는 각
모서리에 두 줄씩 더한 32\*32 크기의 입력을 받는다. 이는 local feature가
이미지의 가장자리에 위치할 수 있기 때문이다. 각 레이어를 순서대로
설명하자면, **첫번째 레이어는 Convolutional 레이어다. 입력 데이터로부터
5\*5 크기의 필터 6개를 통해 28\*28 크기의 feature map 6개를 만든다.
다음으로 sub-sampling 레이어를 지나는데 입력 데이터에 2\*2 filter 6개를
활용해 28\*28 크기의 입력을 14\*14 크기로 변환한다. Average pooling을
사용해 가중치 1개, 편향 1개의 파라미터를 갖고 이를 더한 결과에 sigmoid
함수를 적용한다. 그 다음 레이어는 다시 convolutional 레이어로, 14\*14
feature map 6개에 대해 5\*5 filter 16개를 활용하여 10\*10 feature map
16개를 제작한다. 6개의 입력으로부터 16개의 결과물을 만들 때, 입력들에
대해 선택적으로 연결하여 네트워크의 대칭적인 성질을 없애고자 한다. 이는
convolutional한 결과의 낮은 수준의 feature가 각기 다른 부분과 섞여
global feature를 얻고자 하는 목적이다. 다음으로 sub-sampling layer를
다시 통과한다. 이를 지나가면 10\*10 크기의 16개 입력 데이터가 2\*2
필터를 통해 5\*5 크기의 16개 feature map으로 변형된다. 지난 sub-sampling
layer와 동일하게 가중치와 편향 파라미터를 활용한다. 다음은 다시
Convolutional 레이어다. 5\*5 크기 feature map 16개로부터 5\*5 크기의
filter 120개를 사용해서 1\*1 크기의 feature map 120개를 생성한다. 이전
단계의 16개의 feature map이 서로 섞여 1\*1 feature map 120개가 되는
것이다. 다음은 fully connected 레이어로 1\*1 크기, 120개의 feature
map에서 1\*1 크기, 84개의 feature map을 생성한다. 마지막으로 output
레이어에서는 Euclidean radical basis function을 활용해서 최종 10개의
클래스로 구분한다.**

**이러한 구조를 학습하는 데에 있어 MSE 오차 함수를 활용하였다.**
