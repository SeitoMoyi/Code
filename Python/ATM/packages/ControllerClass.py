


from random import random

from ATM.packages import CardClass
from ATM.packages.UserClass import User


class Controller():
    def register(self):
        user = self.__getuser()
        userid = self.__getuserid()
        userpwd = self.__getuserpwd()
        cardid = self.__getcardid()

        cardid = random.randint(100000,999999)
        cardobj = CardClass.Crad(cardid,userpwd)

        userobj = User.User(user,userid,userpwd,cardid)

    def __getuser(self):
        while True:
            user = input('user')
            if not user:
                print('Error')
                continue
            else:
                return user
    def __getuserid(self):
        while True:
            userid = input('userid')
            if not userid:
                print('Error')
                continue
            else:
                return userid
    def __getuserpwd(self):
        while True:
            userpwd = input('userpwd')
            if not userpwd:
                print('Error')
                continue
            else:
                repwd = input('userpwd')
                if repwd == userpwd:
                    return userpwd
                else:
                    print('Error')
                    continue
    def __getcardid(self):
        while True:
            cardid = input('cardid')
            if not userid:
                print('Error')
                continue
            else:
                return cardid