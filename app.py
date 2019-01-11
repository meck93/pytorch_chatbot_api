from flask import Flask, jsonify, abort, make_response, request
from code.deploy import Deploy

# webapp
app = Flask(__name__, template_folder='./')

# instantiate pre-trained seq2seq model
predictor = Deploy()


@app.route('/test', methods=['GET'])
def get_tasks():
    return jsonify('message': 'server up and running!')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    response = predictor.reply(str(request.json['message']))
    return jsonify(response)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)


if __name__ == '__main__':
    app.run(debug=True)
