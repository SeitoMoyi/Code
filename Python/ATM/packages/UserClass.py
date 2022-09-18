from unicodedata import name


class User():
    def __init__(self,user,userid,phone,card):
        self.user = user
        self.userid = userid
        self.phone = phone
        self.card = card