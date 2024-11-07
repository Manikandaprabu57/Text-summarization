from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Initialize the summarizer pipeline (using PyTorch as the framework)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

# Define a route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the summarization API
@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text input from the frontend
    data = request.json
    text = data.get('text')

    # If no text is provided, return an error
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate a summary
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
