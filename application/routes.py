# classifier/application/routes.py
from flask import request,jsonify
from application import app 
from classifier import classify

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.json
    text = data.get('text') 
    if text is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': 'Parametr "" is invalid'.format(params)}), 400 
    else:
        result = classify(text)
        return jsonify({'result': result})

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'

