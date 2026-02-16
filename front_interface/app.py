from flask import Flask, request, jsonify, render_template
from t_test.testing_img import process_img
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_text():
    data = request.json
    user_text = data.get("text", "")
    
    processed_text = user_text.lower()
    sam_logs = str(process_img(processed_text))
    
    return jsonify({"result": processed_text, "logs": sam_logs})

if __name__ == "__main__":
    app.run(debug=True)
