from socket import IOCTL_VM_SOCKETS_GET_LOCAL_CID


class Crad():
    def __init__(self,cardid,pwd,money = 10,islock = False):
        self.card_id = cardid
        self.password = pwd
        self.money = money
        self.islock = islock