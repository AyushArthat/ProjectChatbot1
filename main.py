from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

app = Flask(__name__)

# Use a more advanced QA model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text content from relevant tags
    texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    content = ' '.join(text.get_text(separator=' ', strip=True) for text in texts)
    
    return content

def clean_content(content):
    # Remove HTML tags and unwanted characters
    content = re.sub(r'<.*?>', ' ', content)
    content = re.sub(r'\s+', ' ', content).strip()
    return content

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scrape_and_answer', methods=['POST'])
def scrape_and_answer():
    url = request.form['url']
    question = request.form['question']
    
    content = scrape_website(url)
    cleaned_content = clean_content(content)
    
    answer = qa_pipeline(question=question, context=cleaned_content)
    
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True)