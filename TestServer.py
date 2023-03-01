from flask import Flask, request

app = Flask(__name__)


@app.route('/results', methods=['POST'])
def handle_results():
    print(request.json)
    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5002)
