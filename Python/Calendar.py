import calendar
def date(year,month):
    res = calendar.monthrange(year,month)
    firstday = res[0]+1
    tot = res[1]
    print(f'{year}year-{month}month calendar')
    print('Mon Tue Wed Thu Fri Sat Sun')
    day = 1
    while day<=tot:
        for i in range(1,8):
            if (i<firstday and day==1) or day>tot:
                print(' '*4,end='')
            else:
                print('{:0>2d}'.format(day),end='  ')
                day+=1
        print()
year = 2022
month = 2
while True:
    date(year,month)
    c = input('< or >')
    if c == '<':
        if month > 1:
            month -= 1
        else:
            year -= 1
            month += 11
    elif c == '>':
        if month < 12:
            month += 1
        else:
            year += 1
            month -= 11
    else:
        quit(0)