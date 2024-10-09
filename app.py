import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
from util.query import query

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def route_query():
    data = request.get_json()
    response = query(data.get('query'))

    if response:
        return jsonify({"message": response}), 200

    return jsonify({"error": "Something went wrong"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
