import os
import sys
import pickle
import string
import pandas as pd

from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support

lemma = WordNetLemmatizer()

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = os.path.basename(database_filepath).split(".")[0]
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    category_names = df.columns[-36:]
    Y = df[category_names]
    return X,Y, category_names


def tokenize(text):
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
    
    # Remove the punctuations
    text = "".join([i for i in text if i not in string.punctuation])
    tokens = word_tokenize(text)
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Lemmatize
    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    return tokens


def build_model():
    # Calculates a vector of term frequencies
    tfidf = TfidfVectorizer()
    # Define MultiOutputClassifier classification
    clf = MultiOutputClassifier(AdaBoostClassifier())
    # Define simple Pipeline
    pipeline = Pipeline([
        ('tfidf',tfidf),
        ('clf',clf)
    ],
    verbose = 1)
    # Use grid search to find better parameters.
    parameters_grid = {'clf__estimator__learning_rate': [0.01, 0.02, 0.05],
              'clf__estimator__n_estimators': [10, 20, 40]}
    pipeline = GridSearchCV(pipeline, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1, verbose = 4)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    P,R,F,_= precision_recall_fscore_support(Y_test,Y_pred)
    results = {}
    for ind, name in enumerate(category_names):
        result = {}
        result['f'] = F[ind]
        result['p'] = P[ind]
        result['r'] = R[ind]
        results[name] = result
    print(classification_report(Y_test.values,model.predict(X_test), target_names = category_names.values))
    return results


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()