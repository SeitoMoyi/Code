class Score():
    def __get__(self,instance,owner):
        return self.__score
    def __set__(self,instance,value):
        if value >= 90 and value <= 100:
            self.__score = value
        else:
            self.__score = 'NMMD'

class Student():
    __score = None
    def gets(self):
        print('GETED')
        return self.__score
    def sets(self,value):
        print('SETED')
        self.__score = value
    def dels(self):
        print('DELED')
        del self.__score
    score = property(gets,sets,dels)
    def __init__(self,id,name,score):
        self.id = id
        self.name = name
        self.score = score
    def ReturnMe(self):
        res = f'''
        学号:{self.id}
        姓名:{self.name}
        分数:{self.score}
        '''
        print(res)

xpb = Student(201850050,'xpb',90)
xpb.score = 80
xpb.ReturnMe()