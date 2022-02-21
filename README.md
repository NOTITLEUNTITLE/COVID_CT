# COVID CT_Image Classification
MNC에서 주관한 흉부 CT 이미지를 이용해 코로나19 감염 여부를 이진 분류 모델 구현 대회입니다.

## 대회 개요
- **대회 기간** : 2022년 02월 09일 12:00 PM ~ 2022년 02월 15일 12:00 PM
- **문제 정의** : 흉부 CT 이미지를 이용해 코로나19 감염 여부를 이진 분류
- **추진 배경**
	- 코로나 펜데믹 도래와 장기화에 따른 코로나 확진자 분류 필요성 증대
	- 의료 이미지 AI 기술 적용 가능성 확대


- **평가 지표** 
	- **Accuracy (정확도)**
![image](https://github.com/NOTITLEUNTITLE/COVID_CT/blob/main/.image1PNG.PNG?raw=true)

----------
## 문제 접근방법
비슷한 유형의 문제를 많이 찾아볼수 있습니다.
[참고 사이트](https://givitallugot.github.io/articles/2021-02/Project-COVID19-CT-Classfication-2)에서 흐름도를 참고해보았습니다.

### 주어진 image가 600개라서 deep learning에 부적합하여, Augmentation을 이용하였습니다.
![image](https://cdn.discordapp.com/attachments/940518751974080532/941562702407548958/unknown.png)

동일한 데이터를 crop하여 학습량을 늘려주었습니다.

### 처음에 무작정 학습을 시켜보니, 학습이 전혀 되지않고 진동을 하여, 사람이 봐도 확진인지 아닌지 분류할수있게 해주면 학습시키는데 더 좋지 않을까하여, 이미지 가공을 하려 했으나 어려움에 부딪히게 되었습니다.   (결국은 아래처럼 성공했습니다.)



![image](https://cdn.discordapp.com/attachments/940518751974080532/941560175989489774/2022-02-11_2.03.52.png)

### 모델 선정
참고한 사이트에서는 cnn을 기반으로 하였으나, 저는 vgg16모델을 사용했습니다.   
이미지분야에서는 Resnet을 사용하면 사실 좋지만, 대회참여와 공부에 더 의미를 두었습니다.

```python
from torchvision.models import vgg16

class  VGG16(nn.Module):
	def  __init__(self, NUM_CLS):
		super(VGG16, self).__init__()
		self.vgg = vgg16(pretrained=False)
		self.features_conv = self.vgg.features
		self.linear = nn.Sequential(
			nn.Linear(73728, 4096),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(4096, NUM_CLS),
			nn.Softmax(dim=1)
		)
	def  forward(self, x):
		x = self.features_conv(x)
		x = torch.flatten(x,1)
		x = self.linear(x)
		return x
```
### 학습결과
```
============================1th fold============================ Loading dataset..
8it [00:04, 1.71it/s]
전체 130개 validation data 중 양성(1) data 62개 
정확도: 0.8385 
정밀도: 0.8361 
재현율: 0.8226 
AUC: 0.8378 
F1: 0.8293 

============================2th fold============================ Loading dataset..
8it [00:04, 1.76it/s]
전체 129개 validation data 중 양성(1) data 52개 
정확도: 0.7907 
정밀도: 0.7119 
재현율: 0.8077 
AUC: 0.7935 
F1: 0.7568

============================3th fold============================ Loading dataset..
8it [00:04, 1.75it/s]
전체 129개 validation data 중 양성(1) data 63개 
정확도: 0.8760 
정밀도: 0.8406 
재현율: 0.9206 
AUC: 0.8770 
F1: 0.8788 

============================4th fold============================ Loading dataset..
8it [00:04, 1.74it/s]
전체 129개 validation data 중 양성(1) data 66개 
정확도: 0.7597 
정밀도: 0.7778 
재현율: 0.7424 
AUC: 0.7601 
F1: 0.7597 

============================5th fold============================ Loading dataset..
8it [00:04, 1.74it/s]
전체 129개 validation data 중 양성(1) data 61개 
정확도: 0.7752 
정밀도: 0.7667 
재현율: 0.7541 
AUC: 0.7741 
F1: 0.7603
```
Early stop()를 작성해서 over fitting이 되면 학습을 종료시켰다.




### ensemble
점수를 올리기 위하여 ensemble을 진행하였으며, soft voting과 hard voting 둘 다 진행하여, 2개를 제출하였습니다.
```python
def  predict(models, loader):
	model1, model2, model3, model4, model5 = models

	 
	file_lst = []
	pred_lst = []
	prob_lst = []
	  
	model1.eval()
	model2.eval()
	model3.eval()
	model4.eval()
	model5.eval()
	  
	with torch.no_grad():
		for batch_index, (img, _, file_num) in tqdm(enumerate(test_dataloader)):
			img = img.to(DEVICE)
			
			prob1 = model1(img)
			prob2 = model2(img)
			prob3 = model3(img)
			prob4 = model4(img)
			prob5 = model5(img)
			  
			prob = (prob1 + prob2 + prob3 + prob4 + prob5) / 5
			file_lst.extend(list(file_num))
			pred_lst.extend(prob.argmax(dim=1).tolist())
			prob_lst.extend(prob[:, 1].tolist())
	return pred_lst, prob_lst, file_lst
```


## 마무리
Resnet 으로 학습을 시켰을경우, 너무 쉽게 over fiting이 발생하였습니다.
over fitting을 방지하기 위하여 4가지를 사용해보았습니다.
-   `Train/Validate/Test` 비율을 `50%/10%/40%`(수치는 선택사항)로 변경하여 `Train` 비율을 낮추고 `Test` 비율을 늘린다.
 -  Dropout 추가 및 Dense Node 줄인다. 
 -  데이터를 증강한다. 
 -  이미지 픽셀을 더 작게 150*150으로 축소한다.
<br><br/><br/><br/>

 아이러니한 상황이죠... 모델이 너무 뛰어나서, 일부러 학습을 더 어렵게하는 경우이죠.
 처음에는 분명히 진동을해서 학습이 안되었는데.....
 


**결론은 Resnet 짱**

	

	
-------
### covid_CT.ipynb : 최종 제출파일
