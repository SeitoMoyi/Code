import imp
from flask import Flask,render_template,request
import pymysql
import time
app = Flask(__name__)

def model(sql):
    db = pymysql.connect(host = 'localhost',
                        user = 'root',
                        password = 'XuPeibin20020509',
                        database = 'test',
                        charset = 'utf8mb4',
                        cursorclass=pymysql.cursors.DictCursor)
    try:
        cursor = db.cursor()
        row = cursor.execute(sql)
        db.commit()
        data = cursor.fetchall()
        if data:
            return data
        else:
            return row
    except:
        db.rollback()
    finally:
        db.close()

@app.route("/")
def hello():
    data = model('select * from message')
    return render_template('index.html',data = data)
@app.route('/add')
def add(): 
    return render_template('add.html')
@app.route('/insert',methods=['POST'])
def insert():
    data = request.form.to_dict()
    data['date'] = time.strftime('%Y-%m-%d %H:%I:%S')
    sql = f'insert into message values(null,"{data["nickname"]}","{data["info"]}","{data["date"]}")'
    res = model(sql)
    print(res)
    if res:
        return '<script>alert("留言成功");location.href="/"</script>'
    else:
        return '<script>alert("留言失败");location.href="/add"</script>'

@app.route('/delete')
def delete():
    id = request.args.get('id')
    sql = f'delete from message where id={id}'
    res = model(sql)
    if res:
        return '<script>alert("删除成功");location.href="/"</script>'
    else:
        return '<script>alert("删除失败");location.href="/"</script>'
if __name__ == '__main__':
    app.run(debug = False, host = '127.0.0.1', port = '8080' ) 