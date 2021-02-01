import flask
import numpy as np
import pickle

model = pickle.load(open('model/titanic_model_classifier.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('main.html'))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {0: 'Dead', 1: 'Survived'}

    return flask.render_template('main.html',
    prediction_text='Passenger will be {}'.format(output[prediction[0]]))

if __name__ == '__main__':
    app.run()
