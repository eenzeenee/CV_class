RBF description

2021312088 백인진

RBF : Radial Basis Function

: 기존 벡터와 입력 벡터의 유사도를 측정하는 함수

분류 문제에서 RBF의 활용

> \- 각 점을 중심으로 임의의 RBF함수를 그린 뒤, 특정한 하나의 클래스의
> 데이터 포인트에 대해 모든 RBF함수를 뒤집을 수 있음
>
> \- 각 지점 x에 RBF함수값을 더하여 전역함수를 제작함.
>
> \- 이렇게 제작한 전역 RBF함수는 중앙 영역에서 밀도가 높고 이외
> 지역에서는 덜 밀집된 모습 보임
>
> \- 이를 통하 저차원 상의 데이터를 고차원으로 사영하여 분류문제에
> 유용하게 사용할 수 있음­­
>
> \- 예시 :
>
> ![](media/image2.png){width="4.571307961504812in"
> height="7.349451006124235in"}
>
> RBFN : Radial Basis Function Networks
>
> ![](media/image1.png){width="5.640625546806649in"
> height="3.682334864391951in"}
>
> \- 2 계층 네트워크에 RBF 뉴런 사용
>
> \- Input layer - radial basis neuron layer - output layer : 매우
> 단순한 구조
>
> \- 각 RBF 뉴런은 중심 벡터를 저장한다. 이때 중심벡터는 훈련
> 데이터셋에서 추출한 하나의 고유한 벡터이다.
>
> \- ![](media/image3.png){width="4.619792213473316in"
> height="1.6345778652668417in"}
>
> \- 각 입력 벡터들은 중심 벡터와 비교되고 그 차이가 RBF 함수에
> 연결된다.
>
> \- 예를 들어, 중심 벡터와 입력 벡터가 동일하면 차이는 0이되고 이를 RBF
> 함수에 연결하면 x=0에서의 정규분포값을 따라 출력은 1이 된다. 이는 해당
> 함수에서의 최고점을 의미한다.
>
> \- 즉 중심 벡터는 RBF의 최고점을 출력하는 값이므로 RBF함수의 중심에
> 있는 벡터이다.
>
> \- 중심 벡터와 입력 벡터의 차이가 커질수록 RBF 뉴런의 출력은 0에
> 가까워진다. 이로 인해 RBF 뉴런은 입력 벡터와 중심 벡터의 유사성에 대한
> 비선형 측정값으로 간주할 수 있다. 이때, RBF 뉴런은 방사형으로
> 나타나므로 (반경 기반 측정) 차이의 상대적인 크기가 중요하다.
>
> \- RBF 노드에서 가중치를 학습하는 데에 있어 출력 노드는
> classification에 중요성을 갖는 RBF 뉴런에 큰 가중치를 제공하고 그렇지
> 않을 경우 작은 가중치를 제공한다.
>
> \- 이러한 네트워크는 그 구조는 단순하나, 매우 복잡한 비선형 모델을
> 구성할 수 있다는 장점을 갖는다.

참고한 사이트 :
https://ichi.pro/ko/bangsahyeong-gijeo-hamsu-rbf-keoneol-mich-rbf-neteuwokeuga-gandanhage-seolmyeong-doem-59039807747948
