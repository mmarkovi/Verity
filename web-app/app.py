from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import json

from model_lib import SimpleNeuralNet, TwoHiddenLayerNeuralNet, SimpleGeneralNeuralNet, load_covid_model, load_fnn_model, load_general_model, predict_model, covid_general_predict_model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predict', methods=["GET"])
def get_prediction():
	title = request.args.get('title')
	content = request.args.get('content')

	lowercased_title = title.lower()
	lowercased_text = content.lower()

	raw_text = lowercased_title + ' ' + lowercased_text

	model_covid, vec_covid = load_covid_model()
	model_fnn, vec_fnn = load_fnn_model()
	model_general, vec_general = load_general_model()

	covid_general_predict, prob_false = covid_general_predict_model(model_general, vec_general, raw_text)

	if covid_general_predict == 1:
		output_prob = predict_model(model_covid, vec_covid, raw_text)
		if output_prob < 0.5:
			prediction = 0
		else:
			prediction = 1
	else:
		output_prob = predict_model(model_fnn, vec_fnn, raw_text)
		if output_prob < 0.5:
			prediction = 0
		else:
			prediction = 1

	if prediction == 0:
		return json.dumps({"prob": output_prob, "prediction": False})
	return json.dumps({"prob": output_prob, "prediction": True})

@app.route('/about')
def about():
    return "About"

if __name__ == "__main__":
    app.run(debug=True)
