class Outer():
    def __call__(self,cls):
        self.cls = cls
        return self.inner
    def inner(self):
        self.cls.name = 'xpb'
        self.cls.func2 = self.func2
        return self.cls
    def func2(self):
        print('func2')  

@Outer()
class love():
    def func1(self):
        print('func1')
obj = love()
obj = obj()
obj.func1()
obj.func2()
print(obj.name)