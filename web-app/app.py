from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

from model_lib import SimpleNeuralNet, load_model, predict_model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/', methods=['POST'])
def form_post():
    title = request.form['title']
    content = request.form['content']
    
    lowercased_title = title.lower()
    lowercased_text = content.lower()

    raw_text = lowercased_title + ' ' + lowercased_text

    model, vec = load_model()
    prediction, prob_false = predict_model(model, vec, raw_text)

    if prediction == 0:
        return "The article might contain false information."
    return "We believe the provided article is true."

# @app.route('/predict')
# def prediction_api():
# 	return "API for prediction"

@app.route('/predict', methods=["GET"])
def get_prediction():
	title = request.args.get('title')
	content = request.args.get('content')

	print('data:', title)
	
	return title + " " + content

@app.route('/about')
def about():
    return "About"

if __name__ == "__main__":
    app.run(debug=True)