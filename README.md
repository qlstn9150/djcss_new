# Deep Joint Source-Channel Coding

#### 진행상황(21.11.15)
- model_CompRatio_SNR
  - basic(0.06, 0.26, 0.49)_(10, 20)
  - model1_(0.06, 0.26)_(10)
  - model2_(0.06, 0.26, 0.49)_(10)
  
- parameter
  - epochs = 5
  - batch size = 16  
    

- 성능 평가1 (압축률에 따른 PSNR)  
![plot1](plot/comp_%5B'basic',%20'model1',%20'model2'%5D_CompRatio%5B0.06,%200.26,%200.49%5D_SNR%5B10%5D.png)

- 성능 평가2 (k/n=0.06)  
![plot2](plot/test_%5B'basic',%20'model1',%20'model2'%5D_CompRatio0.06_SNR%5B10%5D.png)

- 성능 평가2 (k/n=0.26)  
![plot2](plot/test_%5B'basic',%20'model1',%20'model2'%5D_CompRatio0.26_SNR%5B10%5D.png)

- 성능 평가2 (k/n=0.49)  
![plot2](plot/test)

### 향후 계획
- 기존 논문 모델 실행
- 다른 구조 모델 실행해서 비교하기
- 학습 반복 수 늘리기
- 다른 잡음 채널에서 적용하기



#### Requirements
* python == 3.6
* tensorflow == 1.15.0
* keras == 2.3.1
* h5py == 2.10.0

