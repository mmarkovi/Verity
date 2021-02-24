from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/', methods=['POST'])
def form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

@app.route('/about')
def about():
    return "About"

if __name__ == "__main__":
    app.run(debug=True)