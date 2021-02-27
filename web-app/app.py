from flask import Flask, request, render_template
from model_lib import load_model, predict_model

app = Flask(__name__)

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
    predict_model(model, vec, raw_text)

    return lowercased_title + ' ' + lowercased_text

@app.route('/about')
def about():
    return "About"

if __name__ == "__main__":
    app.run(debug=True)