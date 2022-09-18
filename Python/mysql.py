from contextlib import nullcontext
from sqlite3 import SQLITE_PRAGMA
from zlib import Z_SYNC_FLUSH
import pymysql
db = pymysql.connect(host='localhost',
                             user='root',
                             password='XuPeibin20020509',
                             database='test',
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor)
try:
    cursor = db.cursor()
    sql = 'insert into user values(null,"zs","男",2);'
    cursor.execute(sql)
    db.commit()
    data = cursor.fetchone()
except:
    db.rollback()
finally:
    db.close()
