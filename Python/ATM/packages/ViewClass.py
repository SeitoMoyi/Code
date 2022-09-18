import time
class Views():
    def __init__(self):
        self.__showindex()
        time.sleep(1)
        self.showfunc()
    def __showindex(self):
        varstr = '''
***********************
*   Welcome to Bank   *
***********************
        '''
        print(varstr)
    def showfunc(self):
        varstr = '''
***********************
*   1 注册     2 查询  *
***********************
        '''
        print(varstr)

if __name__ == '__main__':
    Views()