import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
# from tensorflow.keras.models import Sequential
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Embedding
from tensorflow.python.keras import preprocessing
import matplotlib.pyplot as plt
# import keras
# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Embedding
# import matplotlib.pyplot as plt

# 配置
n_epoch = 5,
batch_size = 32,
sequence_length = 100,
num_hidden = 100,
num_layer = 3,
num_classes = 2,
non_static = True,
# project_name = sys.argv[1]
project_name = "httpclient"
# project_name = "jackrabbit"
def split_data(data):
    '''将数据中的句向量和标签取出'''
    s_data, l_data = [], []
    for d in data:
        s_data.append(d[1])
        l_data.append(d[2])
    s_data = np.asarray(s_data, dtype="int64")
    l_data = np.asarray(l_data, dtype="int64")
    return s_data, l_data

'''加载训练、验证、测试数据'''
print("load data")
train_data = pickle.load(open(project_name + "/train_nn.pkl", "rb"))
valid_data = pickle.load(open(project_name + "/valid_nn.pkl", "rb"))
test_data = pickle.load(open(project_name + "/test_nn.pkl", "rb"))
print("train data :", len(train_data))
print("valid data :", len(valid_data))
print("test data :", len(test_data))

'''加载句向量中每一个词的index对应的向量'''
#不使用预训练词向量
index2vec = pickle.load(open(project_name + "/index2vec.pkl", 'rb'))
#使用预训练词向量
# index2vec = pickle.load(open(project_name + "/index2vec_pt.pkl", 'rb'))
index2vec = np.asarray(index2vec, dtype='float32')

'''分离数据中的句子和标签'''
train_sentence, train_label = split_data(train_data)
valid_sentence, valid_label = split_data(valid_data)
test_sentence, test_label = split_data(test_data)

print(type(train_sentence))
print(type(train_sentence[0]))
print(train_sentence[0])
print(train_label[0])
print(valid_label[0])
# train_label_ohe = np.eye(2)[train_label] #0->[1. 0.] 1->[0. 1.]
# n_train_batches = train_sentence.shape[0] // batch_size + 1
# n_valid_batches = valid_sentence.shape[0] // batch_size + 1
# n_test_batches = test_sentence.shape[0] // batch_size + 1

'''构建模型'''
print("build the model")
model = Sequential()
model.add(Embedding(9482, 8, input_length=100)) #input=sequence_length error
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(train_sentence, train_label,
                    epochs=n_epoch,
                    batch_size=batch_size,
                    validation_data=(valid_sentence, valid_label))

results = model.evaluate(test_sentence, test_label)
print(results)

# 绘制结果
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()