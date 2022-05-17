컴퓨터 비전 2주차 논문 요약 - FAST

2021312088 백인진

해당 논문은 이미지에서 특징을 검출하는 많은 기법들에 대해 실시간적인
특징이 중시될 경우 이를 신속하게 처리하기 위해서 필요한 충분한
처리속도가 나타나지 않는 점에 기반하여 진행된 연구이다. 이는 동영상,
카메라를 통해 입력된 영상에서의 이미지 특징을 신속하게 검출하기 위한
FAST(Feature from Accelerated Segment Test) 알고리즘을 설명한다. 이미지
내 특정 픽셀 p를 고르고 해당 픽셀의 강도를 Ip라고 가정한 뒤 일정 임계치
t를 정한다. 이후 p 주변 16개 픽셀의 원을 대상으로 원 안에 Ip + t보다
밝은 n개의 인접 픽셀이 있거나 Ip -- t보다 어두운 n개의 인접 픽셀이 있는
경우 모서리로 간주한다. 해당 논문에서는 n을 12개로 정한다. 이때, 속도를
개선하기 위해 16개 원에 해당하는 픽셀 중 1, 9, 5, 13만 검사하여 p가
모서리라면 이 중 적어도 3개는 위에 이야기한 모서리 조건에 해당해야 한다.
그렇지 않을 경우 p를 모서리가 아니라고 간주한다. 이러한 방식은 한계가
존재하는데 먼저 n이 12보다 작은 경우 테스트하고자 하는 픽셀 후보들을
많이 제한할 수 없다. 이미지 검사의 효율성이 검사의 순서, 모서리 형태
분포에 의존적이므로 픽셀 선택은 최적의 해결책이 아니다. 또한 속도 개선을
위한 검사의 결과는 버려지며 모서리 주변으로 이외 다중의 특성이
발견되기도 한다. 이러한 한계 중 1, 2, 3 한계의 경우 머신러닝을 활용한
접근법으로 해결하고자 하였고, 마지막 한계점의 경우 Non-Maximal
Suppresion방식으로 해결된다.

머신러닝을 활용한 방식은 다음과 같다. 학습을 위한 이미지셋을 구한 뒤
모든 이미지에 대해 FAST 알고리즘을 적용하여 특징점을 찾아낸다. 이후
특징점들 주변 16개의 픽셀을 벡터로 저장한다. 이후 특징 벡터 16개를 Pd,
Ps, Pb로 구분지어 나타낸다. 특징점이 모서리일 경우 True, 아닐 경우
False로 나타낸 변수 Kp 또한 정의한다. Pd, Ps, Pb 변수를 활용하여
Kp변수에 대한 decision tree 기반의 ID3 알고리즘을 적용한다. Kp에 대한
엔트로피값을 측정하며 후보 픽셀이 모서리인지 아닌지를 결정하는 데 많은
정보를 제공하는 픽셀 x를 선택한다. 이는 엔트로피가 0이 될때까지
재귀적으로 적용되며 이를 통해 다른 이미지에 대한 빠른 탐지가 가능한
decision tree가 제작된다.

다음으로 Non-Maximal Suppresion(NMS)방식은 다음과 같다. 인접 지역에서
복수 개의 interest point를 찾는 경우에 대한 해결책이다. 먼저 스코어 함수
V를 탐지된 모든 interest point에 대해 계산한다. 이때 V함수는 p픽셀과
이를 둘러싼 16개의 픽셀값 사이 차이의 절대값을 모두 더한 것이다. 이후
두개의 근접한 keypoint에 대해 V함수를 계산한다. 이 중 낮은 V값을 갖는
하나의 keypoint를 제거하여 복수개의 interest point에 대한 문제를
해결하고자 한다.

이러한 방식의 FAST알고리즘은 다른 corner detector에 비해 비교적 빠른
속도를 갖는다는 장점이 있다. 그러나 노이즈의 정도가 심할경우 이에 대해
robust하지 않으며 이러한 결과는 임계값을 어떻게 설정하는지에 따라
달라진다.