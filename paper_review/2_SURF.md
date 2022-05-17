컴퓨터 비전 2주차 논문 요약 - SURF

2021312088 백인진

SURF(Speeded Up Robust Features)는 여러 이미지로부터 스케일, 조명,
시점이 변화하는 것에 관계없이 불변하는 이미지의 feature를 찾아내는
알고리즘 중 하나로 일반적으로 성능이 우수하다고 알려진 SIFT와 유사한
성능을 보이며 더불어 연산 속도를 개선시킨 알고리즘이다. 그러나 이러한
특징을 갖는 SURF는 회색조 이미지 정보만 이용하므로 다양한 색조를 갖는
이미지에 나타나는 feature들을 활용하지는 못한다.

SURF는 속도 개선을 위해 3가지 방법을 제안한다. 첫번째 방법은 Integral
image를 활용하는 것이다. SIFT와 다르게 SURF에서는 integral 이미지를
이용하여 feature를 찾는다. 이때 feature를 찾기 위해 Fast Hessian
Detector를 사용한다. 영상을 적분하여 영상이 갖는 영역의 넓이, 즉 밝기의
합을 구해 추론한다. 적분 영상을 만들어두고 영상의 부분합을 구하여 빠른
속도로 계산할 수 있다. 다음으로 계산 복잡성을 줄이기 위해 간단히 한
detector와 descriptor를 사용한다. 해당 detection을 위해서는 Hessian
Matrix기반으로 detection을 한다. Hessian Matrix를 사용할 경우 정확성이
좋으며 determinant가 최대인 위치에서 blob 구조를 추출할 수 있다. 또한
계산의 단순화를 위해 근사한 Dxx, Dyy 박스 필터를 활용한다. 이를 통해
영역의 넓이를 빠르게 구하고 구한 넓이를 Hessian Matrix로 계산하여
interest point를 추출한다. 또한 SURF도 SIFT와 같이 이미지의 여러
스케일에서 interest point를 추출한다. SIFT의 경우 필터의 크기를 고정하고
이미지의 크기를 줄여가며 interest point를 추출하는 반면, SURF는 이미지의
크기를 고정하고 필터의 크기를 키워가며 interest point를 추출한다. 이는
비교적 계산이 효율적인데, 적분 이미지를 사용하기 때문에 위치를 변경하는
것으로 필터의 크기를 조절할 수 있기 때문이다. 이렇게 추출한 feature 중
극대값을 선정하여 보간법을 적용해 보간된 interest point를 특징 벡터로
저장하여 사용한다. 최종 보간된 interest point를 중심으로 6 scale안의
이웃에 대해 Haar wavelet response를 계산한다. Response의 합을 계산하여
벡터를 생성하고 그 길이가 가장 긴 것이 orientation으로 할당된다. 이 때
Haar wavelet filter를 이용하기에 빠른 계산이 가능하다. 마지막으로
Constrast를 이용하여 빠른 매칭을 진행한다. Hessian matrix를 계산한
라플라시안 부호를 비교하여 비교적 간단한 방식으로 매칭을 시도한다.

이러한 과정을 거쳐 기존 이미지에서 feature를 추출하던 방식들에 비해 보다
빠른 속도로 유사한 성능을 갖는 feature 추출 알고리즘을 제작하였다.
