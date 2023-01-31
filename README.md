# EFL_ContrastiveLearning

## Requirements

```pip install -r requirements.txt```

## Data
- CS쉐어링 콜센터 데이터
- 상담사와 고객간의 통화내역 음성파일을 STT 처리하여 텍스트로 변환
- 위치 : 18서버 `/home/heerak_son/workspace/EFL_ContrastiveLearning/data`
- Data 구조
  - raw(통화내용 STT 텍스트 파일)
  - kss_splitted_data(kss 라이브러리 활용한 문장 분리 시킨 텍스트 파일)
  - preprocessed(데이터 버전별 전처리 된 일반/불만 데이터)
  - cs_sharing(데이터 버전별 배송/제품/처리/기타 데이)
  - 문장감성분석및태깅 50000건.csv(라벨링 맡긴 데이터)
  - 학습데이터셋(최종 학습 파일)
    - sentiment_train.csv
    - sentiment_valid.csv
    - sentiment_test.csv
    - category_train
      - shipping.txt
      - product.txt
      - processing.txt
      - etc.txt
    - category_valid
      - shipping.txt
      - product.txt
      - processing.txt
      - etc.txt
    - category_test
      - shipping.txt
      - product.txt
      - processing.txt
      - etc.txt

## Experiment result

`CS쉐어링 실험 결과 및 데이터 버전관리.xlsx` 이 파일에 기록

## How to Train

- 일반/불만 이진 감성 분류 모델  
  `cs_shharing_sentiment_run.sh`
- 배송/제품/처리/기타 카테고리 분류 모델  
  `cs_sharing_category_run.sh`

## Reference

- [EFL](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot/efl)
- [EFL paper](https://arxiv.org/pdf/2104.14690v1.pdf)
