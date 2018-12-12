import os
import json

from flask import Flask, jsonify, abort, make_response, request


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

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

    @app.errorhandler(404)
    def not_found(error):
        return make_response(jsonify({'error': 'Not found'}), 404)

    @app.route('/model', methods=['POST'])
    def reply():
        if not request.json or not 'title' in request.json:
            abort(400)

        task = {
            'id': tasks[-1]['id'] + 1,
            'title': request.json['title'],
            'description': request.json.get('description', ""),
            'done': False
        }
        tasks.append(task)
        return jsonify({'task': task}), 201

    return app
