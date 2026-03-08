# 导入 torch 库
import torch
import numpy as np
import random

'''
# 打印 PyTorch 版本（确认版本是否正确）
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用（如果有 NVIDIA 显卡）
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

# 测试基础功能（创建张量）
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print(f"张量相加结果: {x + y}")

'''

'''
data = np.random.randn(2, 3)
print(f'np:{data}')
data = torch.tensor(data)
print(data)
'''
# 创建张量
def test01():
    data = torch.IntTensor([2.9, 3.3])
    print(data)
    data = torch.ShortTensor([2.9, 3.3])
    print(data)
    data = torch.LongTensor([2.9, 3.3])
    print(data)
    data = torch.FloatTensor([2.9, 3.3])
    print(data)
    data = torch.DoubleTensor([2.9, 3.3])
    print(data)
# 创建线性空间的张量
def test02():
    #在指定区间按照步长生成元素[start, end, step]
    data = torch.arange(0, 10, 3)
    print(f'arange:{data}')
    #在指定区间按照元素个数生成：0~10之间等分成8份
    data = torch.linspace(0, 10, 8)
    print(f'linspace:{data}')

    data = torch.randn(2, 5)
    print(f'线性张量:{data}')

    data2 = torch.random.initial_seed()
    print(f'随机种子:{data2}')

def test03():
    torch.random.manual_seed(100)
    data = torch.randn(2, 3)
    print(f'随机种子：{data}')
    data2 = torch.randn(2, 3)
    print(f'随机种子：{data2}')
    #检查两个随机种子是否一致
    print(f'data1和data2是否一致：{torch.equal(data, data2)}')
    torch.random.manual_seed(100)
    data3 = torch.randn(2, 3)
    print(f'随机种子：{data3}')
    #检查两个随机种子是否一致
    print(f'data1和data3是否一致：{torch.equal(data, data3)}')

if __name__ == '__main__':
    test03()
    