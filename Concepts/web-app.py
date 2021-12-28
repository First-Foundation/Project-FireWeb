from flask import Flask, redirect, url_for, render_template, request
from flask.templating import render_template_string
from networkx.algorithms import summarization
from forms import SearchForm
import os
from dotenv import load_dotenv
from FireWeb import FireWeb

from googlesearch import search
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from transformers import pipeline

#load .env file
load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')

app = Flask(__name__)
app.config
app.config['SECRET_KEY'] = SECRET_KEY


#Base web page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search", methods=['GET', 'POST'])
def search_results():
    form = SearchForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        
        fw_query = FireWeb(query=None)
        # Save query results
        results = []
        for j in search(query=form.searchfield.data, tld="com", num=10, stop=10, pause=1, tbs="qdr:w"):
            results.append(j)
        print(results)

        text_repo = []
        one_text = """"""
        for url in results:
            paragraph_data = fw_query.extract_data_from_link(url)
            text_repo.append(paragraph_data)
            one_text += paragraph_data

        summary = fw_query.generate_summary(one_text, 5)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        summarizer = pipeline("summarization")

        summary_text = summarizer(summary, max_length=1000, min_length=50)[0]['summary_text']
        #summary_text = summarizer(extractive_summary, max_length=1000, min_length=5, truncation=True)[0]['summary_text']

        return render_template('search.html', summary_text=summary_text, form=form)
    else:
        return render_template('search.html', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)