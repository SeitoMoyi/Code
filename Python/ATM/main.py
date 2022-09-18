from packages.ControllerClass import Controller
from packages.ViewClass import Views
class Main():
    def __init__(self):
        view = Views()
        obj = Controller()
        while True:
            num = input('service')
            if num == '1':
                obj.register()

if __name__ == '__main__':
    Main()