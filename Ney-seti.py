Python 3.8.6 (tags/v3.8.6:db45529, Sep 23 2020, 15:37:30) [MSC v.1927 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
>>> os.chdir ('D:\\NeuralNetwork\\Network1')
>>> import mnist_loader
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
>>> import network
Сеть net:
Количетво слоев: 3
Количество нейронов в слое 0 : 2
Количество нейронов в слое 1 : 3
Количество нейронов в слое 2 : 1
W_ 1 :
[[-0.05  0.31]
 [-0.44 -0.31]
 [ 0.84 -0.72]]
b_ 1 :
[[-1.26]
 [-1.99]
 [ 0.28]]
W_ 2 :
[[ 0.49 -0.51  0.58]]
b_ 2 :
[[-1.62]]
>>> net = network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
Epoch 0: 8227 / 10000
Epoch 1: 8899 / 10000
Epoch 2: 8940 / 10000
Epoch 3: 8751 / 10000
Epoch 4: 8967 / 10000
Epoch 5: 9079 / 10000
Epoch 6: 9035 / 10000
Epoch 7: 9040 / 10000
Epoch 8: 9091 / 10000
Epoch 9: 9014 / 10000
Epoch 10: 9148 / 10000
Epoch 11: 9119 / 10000
Epoch 12: 9047 / 10000
Epoch 13: 9112 / 10000

Warning (from warnings module):
  File "D:\NeuralNetwork\Network1\network.py", line 19
    return 1.0/(1.0+np.exp(-z))
RuntimeWarning: overflow encountered in exp
Epoch 14: 8989 / 10000
Epoch 15: 9110 / 10000
Epoch 16: 9128 / 10000
Epoch 17: 9200 / 10000
Epoch 18: 9129 / 10000
Epoch 19: 9125 / 10000
Epoch 20: 9136 / 10000
Epoch 21: 9248 / 10000
Epoch 22: 9154 / 10000
Epoch 23: 9154 / 10000
Epoch 24: 9070 / 10000
Epoch 25: 9218 / 10000
Epoch 26: 9153 / 10000
Epoch 27: 9179 / 10000
Epoch 28: 9167 / 10000
Epoch 29: 9282 / 10000
>>> 