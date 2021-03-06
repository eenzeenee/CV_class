컴퓨터 비전 3주차 논문 요약 -- Deep learning

2021312088 백인진

먼저 본 논문은 머신러닝의 한계를 지적하며 시작한다. 이는 학습에 사용할
데이터를 알고리즘의 형식에 맞도록 수정해주어야 한다는 점이다. 이와 달리
Representation 학습은, 알고리즘이 데이터를 그대로 입력받고 자동적으로
필요한 표현을 찾아낸다는 점에서 데이터 수정이 필요없다는 장점을 갖는다.
딥러닝은 이러한 representation 학습을 여러단계로 진행하는 것을 말하며,
각 단계의 representation을 비선형 함수 모듈을 통해 더욱 추상화된 다음
단계 레이어의 representation으로 변환한다. 이때 핵심은 이러한 레이어를
사람이 직접 만들어주는 것이 아니라, 일반적인 학습 과정을 통해 데이터에서
스스로 학습해간다는 점이다.

또한 딥러닝 뿐만 아니라 머신러닝에서도 일반적인 학습의 형태는
지도학습이다. 지도학습은 학습 데이터와 출력 데이터를 한쌍으로 활용한다.
모델을 학습하기 위해 주어진 정답과 모델을 통해 도출한 정답 간의 오차를
측정하는 목적함수를 활용하고 이러한 오차를 바탕으로 모델 내부의 가중치를
변경한다. 가중치를 변경하기 위해 학습 알고리즘이 기울기를 계산해내고
이는 벡터로 표현되며 각각의 가중치가 약간씩 변경된다면, 전체 오차에 있어
어떠한 영향을 미치는지를 나타내게 된다. 기울기 벡터의 반대로 가중치를
조절하며 오차를 줄여나갈 수 있다.

SGD는 전체 데이터 중 일부만을 무작위로 골라내어 결과값을 도출하고 오차를
계산하는 방식이다. 이때 기울기의 평균값을 활용하여 가중치를 조정하는
데에 활용된다. 이러한 과정을 반복하여 목적함수의 평균이 더이상 감소하지
않을때까지 반복한다. 이렇게 학습한 모델에 대해 새로운 데이터를 활용하여
모델의 성능을 평가한다.

일반적인 머신러닝에서는 사용자가 임의로 중요한 특징을 추출하는 과정이
필요한데 이를 자동으로 할 수 있는 것이 딥러닝 기법이다. 딥러닝 구조는
학습 레이어를 중첩하여 쌓은 구조로 비선형적인 출력으로 변환하는 과정을
거친다. 각 레이어는 중요 부분을 더욱 강조하고 중요하지 않은 부분은
무시하기 때문에 해결하고자 하는 문제에 대해 중요한 특징에 민감하게
반응하게 된다.

이 과정에서 연쇄법칙을 활용하여 모델의 가중치를 변화시키는 과정을
거친다. 목적함수에 대해 입력에 대한 기울기를 구하기 위해 목적 함수로부터
역순으로 계산을 취할 수 있다는 것인데, 각 레이어에서 나오는 출력이 이전
입력에 대해 얼마나 변화하는지 파악하며 이를 연쇄적으로 곱해가며 최초
입력으로부터 최종 입력에 대한 목적함수의 기울기를 구할 수 있게 된다.

다음으로 CNN을 설명한다. Convolution은 합성곱으로 곱한 것을 다시
더한다는 의미를 갖는다. 이는 다차원의 데이터를 처리하기 위해 등장했으며
Convolution Layer와 Pooling layer로 구성된다. 전자는 특징맵으로 구성되어
특징을 추출해내는 단계이고, 후자는 추출한 특징을 압축하는 단계이다. 이
과정을 통해 찾아내고자 하는 특징의 위치, 모양이 약간 변하더라도 문제
없이 특징을 검출해낼 수 있다. 이렇게 추출한 결과물을 이용하여 이미지를
분류하는 등의 역할을 수행할 수 있다.

다음으로 RNN은 순서적인 특징을 갖는 데이터를 처리하는 데에 특화된
모델이다. 입력에 대해 바로 출력을 내는 것이 아니고 모델 내부에 hidden
state에 저장한 후 이를 각 입력에 대해 계속 쌓아가며 최종적인 결과를
도출한다. 즉, 마지막 단계의 hidden state에는 모든 입력의 정보가 포함되어
있다고 볼 수 있다.

딥러닝은 앞으로 정답이 없는 데이터에 대해 학습할 수 있는 방법인
Unsupervised Learning방식이 대두될 것으로 보이며 이 중요성 또한 커질
것으로 보인다. 더불어 자연어 처리와 관련한 개선 또한 보일 것으로
기대한다.
