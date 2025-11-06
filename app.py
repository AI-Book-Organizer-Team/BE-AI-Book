from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')  # '/' 경로 접속 시 start 실행 (라우팅 이라고 부름)
def start():  # 함수의 이름은 중복만 되지 않으면 됨
    return "Hello World"

if __name__ == '__main__':
    app.run()  # app 실행