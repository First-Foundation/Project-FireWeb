import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TRANSFORMERS_CACHE'] = "c:\\huggingface\\.cache\\"

import requests
import spacy
from googlesearch import search
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from transformers import pipeline

class FireWeb():
    def __init__(self, query):
        self.query = query
    
    ### Helper Functions

    # Open Url and parse with bs4
    def parse_url(self, url):
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'html.parser')
        return soup

    # Extract all paragraph elements from bs4 parsed soup
    def extract_paragraphs(self, parsed_soup):
        text = ""
        for para in parsed_soup:
            text += para.get_text()
        return text

    # Extracts alls paragraph data from a url
    def extract_data_from_link(self, url):
        soup = self.parse_url(url)
        text = self.extract_paragraphs(soup('p'))
        return text

# This is model 1

    def read_data(self, data):
        article = data.split(". ")
        sentences = []
        
        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 
        
        return sentences

    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary(self, word_data, top_n=5):
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  self.read_data(word_data)
        #print(sentences)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        #print("Summarize Text:\n", ". ".join(summarize_text))
        return summarize_text

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ## Setting to use the bart-large-cnn model for summarization
    summarizer = pipeline("summarization")
    #summarizer = pipeline("summarization", model="t5-3b", tokenizer="t5-3b")
    # Search Query
    query = "Cold War"

    # Save query results
    results = []
    for j in search(query, tld="com", num=10, stop=10, pause=1, tbs="qdr:w"):
        results.append(j)
    print(results)

    text_repo = []
    one_text = """"""
    for url in results:
        paragraph_data = FireWeb.extract_data_from_link(url)
        text_repo.append(paragraph_data)
        one_text += paragraph_data
    #print(one_text)

    extractive_summary = FireWeb.generate_summary(one_text, 5)
    print(extractive_summary)

    summary_text = FireWeb.summarizer(extractive_summary, max_length=1000, min_length=50)[0]['summary_text']
    #summary_text = summarizer(extractive_summary, max_length=1000, min_length=5, truncation=True)[0]['summary_text']
    print(summary_text)

if __name__ == '__main__':
    main()