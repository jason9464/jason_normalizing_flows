# jason_normalizing_flows

Todo
1. 데이터 받아오기
  MNIST, TFD, SVHN, CIFAR-10 받아오기
2. 데이터 텐서화
  받아온 데이터 1 dimensional tensor화 시키기
3. preprocessing 시키기
  MNIST: None
  TFD: Approx, whitening
  SVHN, CIFAR-10: ZCA
4. 모델 구성
  MNIST: 1000 units, 5 hidden layers
  TFD: 5000 units, 4 hidden layers
  SVHN: 2000 units, 4 hidden layers
  CIFAR-10: 2000 units, 4 hidden layers
5. 손실 함수, 최적화 구현
  Loss function
    MNIST: logistic
    TFD: gaussian
    SVHN: logistic
    CIFAR-10: logistic
  Optimization
    AdaM
