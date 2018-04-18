# Flask App on AWS EC2 for Solving Machine Learning Problem


### Problem Statement

We process documents related to mortgages, aka everything that happens to originate a mortgage that you don't see as a borrower. Often times the only access to a document we have is a scan of a fax of a print out of the document. Our system is able to read and comprehend that document, turning a PDF into structured business content that our customers can act on.

This dataset represents the output of the OCR stage of our data pipeline. Since these documents are sensitive financial documents we have not provided you with the raw text that was extracted. Instead we have had to obscure the data. Each word in the source is mapped to one unique value in the output. If the word appears in multiple documents then that value will appear multiple times. The word order for the dataset comes directly from our OCR layer, so it should be _roughly_ in order.

Here is a sample line:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

The first field is the document label. Everything after the comma is a space delimited set of word values.

The dataset is included as part of this repo.

### Completed: Flask WebApp (deployed on AWS EC2)

- Trained and pickled document classification models (multiple ensemble models and 1 Keras LSTM model).
```
Ensemble Models:
Different models were trained including: Logistic Regression, Ridge Regression RandomForest, SVM, Naive Bayes, etc.
Three models with better overall performance were picked and combined as our ensemble model.
{'Logistic Regression': {'accuracy': 0.85280248633586964,
  'f1': 0.66120911770769764,
  'model': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
            verbose=0, warm_start=False)},
 'Ridge Classifier': {'accuracy': 0.87337905905047686,
  'f1': 0.73661431176238257,
  'model': RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
          max_iter=None, normalize=False, random_state=None, solver='auto',
          tol=0.001)},
 'SVM classifier': {'accuracy': 0.87568320651591469,
  'f1': 0.75454958827038499,
  'model': LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
       intercept_scaling=1, loss='squared_hinge', max_iter=1000,
       multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
       verbose=0)}}
1.
Training: Logistic Regression
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
Accuracy: 0.85280249
                         precision    recall  f1-score   support

   DELETION OF INTEREST       0.83      0.25      0.38        60
         RETURNED CHECK       0.88      0.91      0.89      5717
                   BILL       0.00      0.00      0.00        82
          POLICY CHANGE       0.80      0.90      0.85      2652
    CANCELLATION NOTICE       0.80      0.85      0.82      2948
            DECLARATION       0.94      0.80      0.86       257
     CHANGE ENDORSEMENT       0.69      0.11      0.19       282
     NON-RENEWAL NOTICE       0.91      0.86      0.89      1433
                 BINDER       0.92      0.57      0.71       237
   REINSTATEMENT NOTICE       0.90      0.13      0.22        71
      EXPIRATION NOTICE       0.98      0.62      0.76       202
INTENT TO CANCEL NOTICE       0.83      0.83      0.83      3173
            APPLICATION       0.94      0.92      0.93      1337
            BILL BINDER       0.99      0.86      0.92       211

            avg / total       0.85      0.85      0.84     18662

[[  15   11    0   24    0    0    0    0    0    0    0   10    0    0]
 [   1 5215    2  136  209    1    0    0    5    0    0  140    8    0]
 [   0   55    0   15    0    0    0    0    0    0    0   12    0    0]
 [   2   78    0 2390    8    3    7    4    0    0    0  159    1    0]
 [   0  228    0   19 2500    0    2   90    1    0    2   65   41    0]
 [   0    3    0    3    5  205    0    0    1    0    0   40    0    0]
 [   0   36    0  138   14    0   31   10    0    0    0   51    2    0]
 [   0    0    0    7  174    1    0 1236    0    0    0   15    0    0]
 [   0   55    0    0   18    0    0    1  136    1    1   18    7    0]
 [   0   20    0    1   37    0    0    1    0    9    0    3    0    0]
 [   0    2    0    0   62    0    0    1    1    0  125   10    1    0]
 [   0  209    0  232   48    7    2   12    3    0    0 2645   14    1]
 [   0   23    0    6   49    0    3    0    1    0    0   27 1227    1]
 [   0    9    0    9    2    0    0    0    0    0    0   10    0  181]]
 
2.
Training: Ridge Classifier
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
        max_iter=None, normalize=False, random_state=None, solver='auto',
        tol=0.001)
Accuracy: 0.87337906
                         precision    recall  f1-score   support

   DELETION OF INTEREST       0.81      0.73      0.77        60
         RETURNED CHECK       0.90      0.91      0.90      5717
                   BILL       0.29      0.07      0.12        82
          POLICY CHANGE       0.82      0.91      0.87      2652
    CANCELLATION NOTICE       0.84      0.88      0.86      2948
            DECLARATION       0.92      0.81      0.86       257
     CHANGE ENDORSEMENT       0.60      0.15      0.23       282
     NON-RENEWAL NOTICE       0.93      0.89      0.91      1433
                 BINDER       0.88      0.71      0.79       237
   REINSTATEMENT NOTICE       0.61      0.27      0.37        71
      EXPIRATION NOTICE       0.95      0.85      0.90       202
INTENT TO CANCEL NOTICE       0.85      0.86      0.85      3173
            APPLICATION       0.94      0.94      0.94      1337
            BILL BINDER       0.97      0.91      0.94       211

            avg / total       0.87      0.87      0.87     18662

[[  44    1    0    9    0    0    0    0    0    0    0    6    0    0]
 [   4 5188   10  115  225    1    8    1    7    6    0  143    9    0]
 [   0   56    6    8    0    0    0    0    0    0    0   12    0    0]
 [   2   64    2 2417    7    5    9    3    0    0    0  142    1    0]
 [   2  185    0   14 2584    0    1   66    1    3    3   43   45    1]
 [   0    2    1    2    4  209    0    0    1    0    0   38    0    0]
 [   2   32    0  134   13    0   41    9    1    1    0   47    2    0]
 [   0    0    0    4  136    1    1 1281    0    0    0   10    0    0]
 [   0   50    0    0    6    0    0    1  169    1    1    9    0    0]
 [   0   16    0    2   29    0    0    1    0   19    1    1    2    0]
 [   0    1    0    0   16    0    1    2    4    0  172    5    1    0]
 [   0  147    2  220   26   11    4   14    8    0    5 2715   17    4]
 [   0   12    0    3   29    0    3    3    1    1    0   21 1263    1]
 [   0    8    0    4    1    0    0    0    0    0    0    7    0  191]]
 
3.
Training: SVM classifier
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
     verbose=0)
Accuracy: 0.87568321
                         precision    recall  f1-score   support

   DELETION OF INTEREST       0.78      0.75      0.76        60
         RETURNED CHECK       0.90      0.90      0.90      5717
                   BILL       0.29      0.10      0.15        82
          POLICY CHANGE       0.83      0.91      0.87      2652
    CANCELLATION NOTICE       0.84      0.88      0.86      2948
            DECLARATION       0.91      0.84      0.88       257
     CHANGE ENDORSEMENT       0.53      0.23      0.32       282
     NON-RENEWAL NOTICE       0.93      0.90      0.92      1433
                 BINDER       0.86      0.73      0.79       237
   REINSTATEMENT NOTICE       0.68      0.37      0.48        71
      EXPIRATION NOTICE       0.95      0.86      0.90       202
INTENT TO CANCEL NOTICE       0.86      0.85      0.85      3173
            APPLICATION       0.94      0.95      0.95      1337
            BILL BINDER       0.97      0.91      0.94       211

            avg / total       0.87      0.88      0.87     18662

[[  45    0    0   10    0    0    0    0    0    0    0    5    0    0]
 [   7 5164   16  106  241    1   21    0    8    6    0  138    9    0]
 [   0   54    8    8    0    0    0    0    0    0    0   12    0    0]
 [   2   58    2 2424    7    4   16    3    0    0    0  135    1    0]
 [   1  179    0   11 2600    0    3   58    2    3    3   41   46    1]
 [   0    1    0    2    3  217    0    1    1    0    0   32    0    0]
 [   2   32    0  117   14    0   66    9    2    1    0   37    2    0]
 [   0    0    0    5  120    1    5 1294    0    0    0    8    0    0]
 [   0   46    0    1    5    0    0    1  173    1    1    8    1    0]
 [   0   14    0    2   25    0    0    1    0   26    1    1    1    0]
 [   0    1    0    0   13    0    2    2    4    0  173    6    1    0]
 [   1  156    2  219   31   16    7   15    9    0    5 2695   13    4]
 [   0   10    0    3   26    0    5    3    1    1    0   21 1266    1]
 [   0    8    0    6    1    0    0    0    0    0    0    5    0  191]]
 
 
Keras LSTM Model:
From confusion matrix above, we can see the BILL and CHANGE ENDORSEMENT labels are hard to predict.
Thus, Keras LSTM model is implemented to handle the 2 special document types:
[RETURNED CHECK, POLICY CHANGE] to ckeck if BILL or CHANGE ENDORSEMENT be misclassified as RETURNED CHECK or POLICY CHANGE

Using TensorFlow backend.
Train on 21863 samples, validate on 9370 samples
Epoch 1
21863/21863 [==============================] - 1010s 46ms/step - loss: 0.3813 - acc: 0.8740 - val_loss: 0.2603 - val_acc: 0.9116
Epoch 2
21863/21863 [==============================] - 962s 44ms/step - loss: 0.2302 - acc: 0.9282 - val_loss: 0.2429 - val_acc: 0.9202
Epoch 3
21863/21863 [==============================] - 860s 39ms/step - loss: 0.1754 - acc: 0.9469 - val_loss: 0.2869 - val_acc: 0.9042
Epoch 4
21863/21863 [==============================] - 1015s 46ms/step - loss: 0.1365 - acc: 0.9614 - val_loss: 0.3029 - val_acc: 0.9153
After 4 epoches, train acc 0.9614, test acc 0.9153
```
- Build Flask REST API for machine learning models.
```
Front-end: JQuery, Bootstrap
Back-end: Flask server
Apache or Nginx with uWSGI can also be used as the web server for Flask
```
- Deployed Flask web app to [AWS EC2](http://54.85.253.80) as a webservice, which supports submitting input words or uploaded .txt file to predict document type. 
```
AWS EC2 details:

Ubuntu Server 14.04 LTS (HVM), SSD Volume Type - ami-38708b45
Instance type: t2.medium
Availability zone: us-east-1
IPv4 Public IP: 54.85.253.80
env: anaconda3 python3.6
************************
[How To Set Up]
sudo apt-get install python-pip, python-dev
...
clone the webapp repo to /var/www/webapp
...
install anaconda3 (need to match pickle version):
wget https://repo.continuum.io/archive/Anaconda3-4.x.x-Linux-x86_64.sh
...
pip install -r requirements.txt
...
export PATH=~/anaconda3/bin:$PATH
cd /var/www/webapp
sudo env "PATH=$PATH" python app.py
* Running on http://0.0.0.0:80/ (Press CTRL+C to quit)
```

