# 导入 torch 库
import torch
import numpy as np
import random


# 创建张量
def test01():
    #全零张量
    data = torch.zeros(2,3)
    print(data)
    print('*'*50)
    #生成2行3列张量，元素值介于1~20之间
    data = torch.randint(1,20, [2, 3])
    print(data)
    print('*'*50)
    # 各元素值+10 原值不变
    new_data = data.add(10)
    print(new_data)
    print(data)
    print('*'*50)
    # add_ 改变原值
    new_data2 = data.add_(10)
    print(new_data2)
    print(data)
# 创建线性空间的张量
def test02():
    data1 = torch.tensor([[1,2], [3,4], [5,6]])
    data2 = torch.tensor([[7,8], [9,10]])
    print(f'data1:{data1}, shape:{data1.shape}')
    print(f'data2:{data2}, shape:{data2.shape}')

# 张量 data.numpy  torch.from_numpy 互换  共享内存
def test03():
    data = torch.tensor([2, 3])
    numpydata = data.numpy()

    print(f'data：{data }')

if __name__ == '__main__':
    test02()
    