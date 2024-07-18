from flask import Flask, request, jsonify
from custom_lib.direct_handler import make_dir, join_path, get_file_list
from custom_lib.object_judgement import load_model_from_checkpoint, run_inference, categorize_scores, get_final_judgment

app = Flask(__name__)

model = load_model_from_checkpoint("ckpt-11")
THRESHOLDS = [0.9, 0.3]

@app.route('/')
def get():  # put application's code here
    return jsonify({"data": "Not implemented"}), 501


@app.route('/api/judge', methods=['POST'])
def post():
    body = request.get_json()
    if "src" not in body:
        return jsonify({"error": "src is required"}), 400
    src = body["src"]
    scores = run_inference(model, src)
    category_scores = categorize_scores(scores, THRESHOLDS)
    judgment = get_final_judgment(category_scores)
    return jsonify({"data": judgment}), 201


if __name__ == '__main__':
    app.run()
