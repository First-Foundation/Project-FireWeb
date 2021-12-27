from flask import Flask, redirect, url_for, render_template, request
from forms import SearchForm
import os
from dotenv import load_dotenv

#load .env file
load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')

app = Flask(__name__)
app.config
app.config['SECRET_KEY'] = SECRET_KEY

#Base web page
@app.route("/")
@app.route("/home")
def home():
    return render_template('base.html')

@app.route("/search", methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if request.method == 'POST' and form.validate_on_submit():
        print(form.searchfield.data)
    return render_template('search.html', title='Search', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)