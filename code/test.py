# f = open("../data/readme.txt",'r')
# string = f.read()
# print(string)
# strline = f.readline()
# print(strline)
# f.close()

# f=open('../data/readme.txt','r')
# strline = f.readline()
# print(strline)
# f.close()

# f = open('../data/readme.txt','w')
# f.write('hello world \nhello')
# f.close()

'''
方法的重写
'''
# class People:
#     def speak(self):
#         print("people is speaking")
# class Student(People):
#     #方法重写，重写父类的ｓｐｅａｋ方法
#     def speak(self):
#         print('student is speaking')
# class Teacher(People):
#     pass

# #Ｓｔｕｄｅｎｔ类的实例
# s = Student()
# s.speak()
# t = Teacher()
# t.speak()

#多态特性
# class Animal:
#     def eat(self):
#         print("animal is eating")
# class Dog(Animal):
#     def eat(self):
#         print("dog is eating")
# class Cat(Animal):
#     def eat(self):
#         print('cat is eating')
# def eatting_double(animal):
#     animal.eat()
#     animal.eat()

# animal = Animal()
# dog = Dog()
# cat = Cat()
# eatting_double(animal)
# eatting_double(dog)
# eatting_double(cat)


'''列表生成式'''
#列表生成式通常是结合range函数一起使用的，所以先了解range函数的使用方法
# r = range(0,4)
# print('r:',r)
# for x in r:
#     print(x,end="")
# b = list(range(0,6,2))
# print('b:',b)
# c = tuple(range(2,9,3))
# print('c:',c)

# #列表生成式
# data = [1,2,3,4]
# def func(x):
#     return x**2
# d = [func(x) for x in data]
# print('d:',d)
# h = ['HD','FASFSA','SDAGSG']
# M = [x.lower() for x in h]
# print(M)

# filter()用于过滤序列，接受两个参数；一个函数和一个序列，将函数作用在序列的每个元素上，根据函数的返回值是ｔｒｕｅ还是ｆａｌｓｅ，
# 来决定是否舍弃该元素，最终返回一个迭代器
def is_even(x):
    return x%2 == 0
l = filter(is_even,[0,1,2,3,4,5])
print(l)
for val in l:
    print(val)
#使用匿名函数的形式来使用ｆｉｌｔｅｒ函数
str_tuple = ('hiasfasda','asfsda','sagdsgfds','rtyhfb','asfdaspython')
result = filter((lambda x:x.find('python')!=-1),str_tuple)
for string in result:
    print('result:',string)