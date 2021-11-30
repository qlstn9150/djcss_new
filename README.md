# Deep Joint Source-Channel Coding

    
## 모델 별 PSNR 비교  
![PSNR](plot/plot1_psnr/%5B'basic',%20'model3',%20'model4',%20'model6',%20'model7'%5D_CompRatio%5B0.06,%200.26,%200.49%5D_SNR%5B10%5D.png)
## 모델 별 SSIM 비교
![SSIM](plot/plot1_ssim/%5B'basic',%20'model3',%20'model4',%20'model6',%20'model7'%5D_CompRatio%5B0.06,%200.26,%200.49%5D_SNR%5B10%5D.png)
## model6 성능 평가 : 압축률에 따른 PSNR  
![plot1](plot/plot1_psnr/%5B'basic',%20'model6'%5D_CompRatio%5B0.06,%200.26,%200.49%5D_SNR%5B0,%2010,%2020%5D.png)
## model6 성능 평가 : k/n=0.49일 때 test SNR에 따른 PSNR  
![plot2](plot/plot2/%5B'basic',%20'model7'%5D_CompRatio0.49_SNR%5B0,%2010,%2020%5D.png)
## Image
- original
![image](img/Figure_1.png)


#### parameter
  - epochs = 5
  - batch size = 16  

#### Requirements
* python == 3.6
* tensorflow == 1.15.0
* keras == 2.3.1
* h5py == 2.10.0

