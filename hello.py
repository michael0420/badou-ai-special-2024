import torch
x = torch.arange(12)
print(x)
X = x.reshape(3, 4)
print(X)
print('北京欢迎你')
print('b')
print(chr(91))
print(chr(91))
print(chr(91))
fp=open('note.txt', 'w')
print('湖南欢迎你', file=fp)
fp.close()
