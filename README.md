# CT 사진을 기반으로 한 폐암예측모델 
### 주제 선정 배경

전공의 사직과 의료진 파업으로 인한 의료 공백이 발생하고 있습니다. 이로 인해 많은 환자들이 치료 기회 박탈 및 건강 위협의 문제점을 겪고 있습니다. 따라서 저희 팀은 폐암 예측 모델 개발을 통해 현 상황에 조금이나마 기여해보고자 해당 주제를 선택하게 되었습니다.
### 기대효과

- 조기 발견 및 진단을 통한 생존율 증가
- 의료진의 진단 과정 지원으로 정확한 진료 가능
- 비용 절감 및 효율적 의료 서비스 제공

## 프로젝트 소개


폐암에도 여러 종류가 있습니다. 먼저, 암세포의 크기가 작은 것을 소세포폐암이라고 하고, 작지 않은 것은 비소세포폐암이라고 합니다. 발생하는 폐암의 80~85%를 차지하는 비소세포암은 편평세포암, 선암, 대세포암 등으로 나뉩니다. 
<br/><br/>
<p align="center"><img src = "https://github.com/user-attachments/assets/353b1fb7-c537-4799-bc1b-a58da2769b1b"></p>

<p align="center">선암종</p>

<p align="center">Adenocarcinoma</p>

<br/>

<p align="center"><img src = "https://github.com/user-attachments/assets/693c221f-a93c-4285-8dba-743f23a3c352"></p>

<p align="center">편평세포암종</p>

<p align="center">Squamous cell carcinoma</p>

<br/>

<p align="center"><img src = "https://github.com/user-attachments/assets/e827d0ba-650e-4b8c-90bc-e6f18c42f8f6"></p>

<p align="center">대세포암종</p>

<p align="center">Large cell carcinoma</p>

## 사용한 모델
**ResNet**은 2015년 ILSVRC(Image Large Scale Visual REcognition challenge) 대회에서 우승을 한 합성곱 신경망 모델입니다.

<br/>

<img src="https://github.com/user-attachments/assets/0ad329c2-6081-4ae1-99ab-17d51d186f8a" width="420" height="400" align="left" style="margin-right: 10;"/>

### Conv 층을 통과한 F(X) 과 Conv 층을 통과하지 않은
### X 을 더하는 과정을 Residual Mapping 이라고 합니다. 
### 위 Residual Block이 여러 개 쌓여서 나온 CNN 모델을 Residual Network(ResNet)이라고 부릅니다. 
### 모델명에 붙은 숫자는 층의 개수를 의미합니다. 
### 즉, ResNet18은 18개의 층이 있다는 소리이고 ResNet34는 34개의 층이 있다는 의미입니다.









