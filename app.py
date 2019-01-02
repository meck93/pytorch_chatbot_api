from flask import Flask, jsonify, abort, make_response, request
from deploy import Deploy

# webapp
app = Flask(__name__, template_folder='./')
predictor = Deploy()

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})


@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]

    if len(task) == 0:
        abort(404)

    return jsonify({'task': task[0]})


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    response = predictor.reply(str(request.json['message']))
    print(response)
    return jsonify(response)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=True)
