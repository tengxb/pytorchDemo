import sys

# print("hello world!")

# 这里是单行注释

'''
你好，这里是多行注释 demo
嘻嘻
不嘻嘻
略略略...


# name='张三'
# age = 18
# print(f"姓名：{name}，今年{age}岁")
# print(type(name))
# print(type(age))


# pwd=input("请随便输入内容：")
# print('你的输入内容：'+pwd)

# age = 100
# if age>= 18:
#     print('OK')
# elif age <= 7:
#     print("ok2")
# else :
#     print(age)

a='nihao'
print(a.ljust(3))
print(len(a.ljust(10)))

'''

lst = [i for i in range(10) if i%2==0]
print(lst)
lst1=[(m, n) for m in range(1,3) for n in range(3)]
print(lst1)


str='sdjfklsd'
print(str)
lst2=list(str)
lst2.remove('f')
str=''.join(lst2)
print(str)

if str == '':
    print("xixi")
    str = 'buxixi'
elif str == 'xx':
    print('buxixi')
else:
    print('oo')
