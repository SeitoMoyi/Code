usernames = []
passwords = []
blacknames = []
def importdata():
    with open('./data.txt','a+',encoding='utf-8') as fp:
        fp.seek(0)
        res = fp.readlines()
        for i in res:
            i = i.strip()
            arr = i.split(':')
            usernames.append(arr[0])
            passwords.append(arr[1])
    with open('./black.txt','a+',encoding='utf-8') as fp:
        fp.seek(0)
        res = fp.readlines()
        for i in res:
            i = i.strip()
            blacknames.append(i)

def register():
    site = True
    while site:
        username = input('Username:')
        if username in usernames:
            print('Username has been used')
        else:
            while True:
                pwd = input('Password:')
                repwd = input('Confirm your password:')
                if pwd == repwd:
                    print('Success')
                    site = False
                    break
    with open('./data.txt','a+',encoding='utf-8') as fp:
        fp.write(f'{username}:{pwd}\n')

def login():
    site = True
    while site:
        username = input('Username:')
        if username in usernames :
            if username not in blacknames :
                inx = usernames.index(username)
                errornum = 3
                while True:
                    pwd = input('Password:')
                    if(pwd == passwords[inx]):
                        print('Successfully login')
                        site = False
                        break
                    else:
                        if(errornum >=1):
                            print(f'You have {errornum} more chances')
                            errornum -= 1
                        else:
                            print('User locked')
                            with open('./black.txt','a+',encoding='utf-8') as fp:
                                fp.write(username+'\n')
                            site = False
                            break
            else:
                print('User locked')
        else:
            print('User inexist')

while True:
    importdata()
    surface = '''
    ***************************************
    ***Register(0) Login(1) Quit(Others)***
    ***************************************
    '''
    print(surface)
    service = input('Choose your service')
    if service == '0':
        register()
    elif service == '1':
        login()
    else:
        exit(0)