class Demo():
    __obj = None
    name = 'xyz'
    def __new__(cls,*args,**kwargs):
        if cls.__obj:
            return cls.__obj
        else:
            obj = object.__new__(cls)
            cls.__obj = obj
            return cls.__obj
a = Demo()
b = Demo()
print(a,b)
print(a.name,b.name)