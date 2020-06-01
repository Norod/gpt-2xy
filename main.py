import os

from flask import Flask, request, send_file
from model import extend


app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_file('favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return send_file('index.html')
    data = request.form.get('text')
    return extend(data)


if __name__ == "__main__":
    app.run('0.0.0.0', os.environ.get('PORT', 8080))
