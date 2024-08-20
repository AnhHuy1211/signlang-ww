import datetime
import time

from flask import Flask, request, jsonify
from custom_lib.direct_handler import make_dir, join_path, get_file_list
from custom_lib.object_judgement import load_model, run_detection, categorize_scores, get_final_judgment, extract_frames
from custom_lib import config_tf as cfg

app = Flask(__name__)

model = load_model(cfg.EXPORTED_MODEL_PATH)
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
    start_time = time.time()
    scores, frames = run_detection(model, src)
    base64_min, base64_avg, base64_max = extract_frames(scores, frames)
    categories = categorize_scores(scores, THRESHOLDS)
    final_judgment, percentage = get_final_judgment(categories)
    end_time = time.time()
    print({
        "date": datetime.datetime.now(),
        "source": src,
        "scores": categories,
        "elapsed_time": end_time - start_time
    })
    return jsonify({
        "data": {
            "judgement": final_judgment,
            "scores": percentage,
            "evidences": [base64_min, base64_avg, base64_max]
        }
    }), 201


if __name__ == '__main__':
    app.run()
