import time
class log():
    name = time.strftime('%Y-%m-%d')
    url = './'
    fp = None

    def __init__(self):
        self.fp = open(self.url+self.name+'.txt','a+',encoding='utf-8')
    
    def wlog(self,s):
        date = time.strftime('%Y-%m-%d %H:%M:%S')
        msg = date+' '+s+'\n'
        self.fp.write(msg)

    def __del__(self):
        self.fp.close

func = log()
func.wlog('Error')