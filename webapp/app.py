import pandas as pd
import numpy as np
import _pickle as cPickle
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from keras.models import load_model
from werkzeug.utils import secure_filename
from wtforms import SubmitField, StringField
from keras.preprocessing import sequence


# UPLOAD_PATH = './upload/'
UPLOAD_PATH = '/var/www/webapp/upload/'  # file path on AWS EC2
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
csrf = CSRFProtect(app)

app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config.update(dict(
    SECRET_KEY="123",
    WTF_CSRF_SECRET_KEY="123"
))


class predict_form(FlaskForm):
    words = StringField("Words")
    submit = SubmitField("Submit")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@csrf.exempt
@app.route('/', methods=['GET', 'POST'])
def index():
    form = predict_form(request.form)
    result = 'Please input words or upload .txt file!'
    if request.method == 'POST':
        words = form.words.data

        try:
            # handle input words
            if len(words.strip()) > 0:
                result = 'Document Label: \"' + predict(words) + '\"'
            # handle uploaded .txt file
            else:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = app.config['UPLOAD_PATH'] + filename
                    file.save(file_path)

                    with open(file_path, 'r') as temp_file:
                        words = temp_file.read()

                    result = 'Document Label: \"' + predict(words) + '\"'

        except:
            # blank or invalid input
            result = 'Cannot predict the class of your document!'

    print(result)
    return render_template('index.html', form=form, prediction=result)


def predict(words):
    x = pd.Series(words)
    x_train = tfidf1.transform(x)
    y = predict_ensemble(x_train)
    if y in special_types:
        try:
            y_lstm = predict_lstm(words)
            if y_lstm in other_types:
                y = y_lstm
        except:
            pass
    return y


def predict_ensemble(x):
    lr = ensemble['Logistic Regression']
    ridge = ensemble['Ridge Classifier']
    svm = ensemble['SVM classifier']

    p_res = lr.predict(x).tolist()[0]
    pac_res = ridge.predict(x).tolist()[0]
    rc_res = svm.predict(x).tolist()[0]

    res = filter_result(p_res, pac_res, rc_res)
    return res


# use Keras LSTM model for 2 special document types
def predict_lstm(x):
    x_train = tfidf2.texts_to_sequences([x])
    x_train = sequence.pad_sequences(x_train, maxlen=150)

    x_train = np.array(x_train)
    x_train = x_train.reshape((1, 150))
    y_pred_prob = lstm_model.predict(x_train)

    index = np.argmax(y_pred_prob)
    y_pred = label_encoder.inverse_transform(index)
    return y_pred


def filter_result(lr, ridge, svm):
    if svm == ridge:
        return svm
    elif svm == lr:
        return svm
    elif ridge == lr:
        return ridge
    else:
        return svm


if __name__ == '__main__':
    with open('./model/ensemble.pkl', 'rb') as file:
        ensemble = cPickle.load(file)

    with open('./model/label_encoder.pkl', 'rb') as file:
        label_encoder = cPickle.load(file)

    with open('./model/tfidf1.pkl', 'rb') as file:
        tfidf1 = cPickle.load(file)

    with open('./model/tfidf2.pkl', 'rb') as file:
        tfidf2 = cPickle.load(file)

    lstm_model = load_model('./model/lstm.h5')
    special_types = ['RETURNED CHECK', 'POLICY CHANGE']
    other_types = ['DELETION OF INTEREST', 'BILL', 'CANCELLATION NOTICE', 'DECLARATION', 'CHANGE ENDORSEMENT',
                   'NON-RENEWAL NOTICE', 'BINDER', 'REINSTATEMENT NOTICE', 'EXPIRATION NOTICE',
                   'INTENT TO CANCEL NOTICE', 'APPLICATION', 'BILL BINDER']

    # app.run(host='localhost')
    app.run(host='0.0.0.0', port=80)
