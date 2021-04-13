import data_preprocessing
import feature_selection
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

#string to test
#BAG OF WORDS

nb_pipeline = Pipeline([
        ('NBCV',feature_selection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(data_preprocessing.train_news['Statement'],data_preprocessing.train_news['Label'])
predicted_nb = nb_pipeline.predict(data_preprocessing.test_news['Statement'])
np.mean(predicted_nb == data_preprocessing.test_news['Label'])


svm_pipeline = Pipeline([
        ('svmCV',feature_selection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(data_preprocessing.train_news['Statement'],data_preprocessing.train_news['Label'])
predicted_svm = svm_pipeline.predict(data_preprocessing.test_news['Statement'])
np.mean(predicted_svm == data_preprocessing.test_news['Label'])


sgd_pipeline = Pipeline([
        ('svm2CV',feature_selection.countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
        ])

sgd_pipeline.fit(data_preprocessing.train_news['Statement'],data_preprocessing.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(data_preprocessing.test_news['Statement'])
np.mean(predicted_sgd == data_preprocessing.test_news['Label'])



def build_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(data_preprocessing.train_news):
        train_text = data_preprocessing.train_news.iloc[train_ind]['Statement'] 
        train_y = data_preprocessing.train_news.iloc[train_ind]['Label']
    
        test_text = data_preprocessing.train_news.iloc[test_ind]['Statement']
        test_y = data_preprocessing.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(data_preprocessing.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
print("Naive Bayes : ")
build_confusion_matrix(nb_pipeline)
print("SVM : ")
build_confusion_matrix(svm_pipeline)
print("SGD : ")
build_confusion_matrix(sgd_pipeline)


#Now using n-grams
#naive-bayes classifier

print("Using tri-grams: \n\n\n")

nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',feature_selection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(data_preprocessing.train_news['Statement'],data_preprocessing.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(data_preprocessing.test_news['Statement'])
np.mean(predicted_nb_ngram == data_preprocessing.test_news['Label'])


#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',feature_selection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(data_preprocessing.train_news['Statement'],data_preprocessing.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(data_preprocessing.test_news['Statement'])
np.mean(predicted_svm_ngram == data_preprocessing.test_news['Label'])


#sgd classifier
sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',feature_selection.tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
         ])

sgd_pipeline_ngram.fit(data_preprocessing.train_news['Statement'],data_preprocessing.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(data_preprocessing.test_news['Statement'])
np.mean(predicted_sgd_ngram == data_preprocessing.test_news['Label'])



print("Naive Bayes : ")
build_confusion_matrix(nb_pipeline_ngram)
print("SVM : ")
build_confusion_matrix(svm_pipeline_ngram)
print("SGD : ")
build_confusion_matrix(sgd_pipeline_ngram)
print("\n\n")
print("Naive Bayes : ")
print(classification_report(data_preprocessing.test_news['Label'], predicted_nb_ngram))
print("SVM : ")
print(classification_report(data_preprocessing.test_news['Label'], predicted_svm_ngram))
print("SGD : ")
print(classification_report(data_preprocessing.test_news['Label'], predicted_sgd_ngram))


