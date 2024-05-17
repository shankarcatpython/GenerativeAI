from flask import render_template, request , jsonify
from app import app
import llm
#import agents as useagents

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_context', methods=['POST'])
def process_context():
    print('accessing process_context')
    context = request.form['context']
    #response = 
    summarize_response = llm.askme_questions_summarize(context)
    suggestive_response = llm.askme_questions_suggestion(context)
    return jsonify({'summarize_response': summarize_response, 'suggestive_response': suggestive_response})