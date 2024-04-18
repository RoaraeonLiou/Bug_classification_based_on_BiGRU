import pickle

path = r'C:\Users\msi\Desktop\bug-classification\dataset/httpclient.pkl'  # pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f, encoding='latin1')
for i in data:
    print(i)
