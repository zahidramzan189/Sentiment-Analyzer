from flask import Flask, render_template, request
from sentiment import analyze_with_vader, analyze_with_bert

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    vader_result = bert_result = ''
    if request.method == 'POST':
        text = request.form['text']
        vader_result = analyze_with_vader(text)
        bert_result = analyze_with_bert(text)
    return render_template('index.html', vader_result=vader_result, bert_result=bert_result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
