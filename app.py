from flask import Flask, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('logit.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    return render_template('returns.html')

if(__name__=="__main__"):
    app.run(debug=True)
