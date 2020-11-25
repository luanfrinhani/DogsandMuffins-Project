 
from flask import Flask, render_template, request, url_for, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

 

@app.route('/')
def homepage():
    return render_template('index.html')
 
learn = load_learner(path='./Model', file='Segmantation-1.pkl')
classes = learn.data.classes
logging.debug('Learnt classes')


def predict_single(img_file):
    '''function to take image and return prediction'''
    logging.debug('Funcao para pegar uma imagem e retornar a prediction')

    prediction = learn.predict(open_image(img_file))
    probs_list = prediction 
    logging.debug('Antes da lista de probabilidades')
    return probs_list

 

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':  
        logging.debug('Antes da predição')
        my_prediction = predict_single(request.files['image'])        
        logging.debug('Antes da precição final')
        final_pred = my_prediction[0]
        logging.debug('Depois da predição final')
    return render_template('results.html', prediction=final_pred,
                           comment='asd')
 
if __name__ == '__main__':
    app.run(debug=True)
