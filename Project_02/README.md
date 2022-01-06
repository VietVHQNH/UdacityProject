# Disaster Response Pipeline Project

# Write-a-Data-Science-Blog-Post
A Udacity Data Scientist Nanodegree Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [How to run](#run)
4. [Files in the repository](#constructure)

## Installation <a name="installation"></a>

1. nltk
2. flask
3. plotly
4. pandas
5. sqlalchemy
6. scikit-learn

## Project Motivation<a name="motivation"></a>

In this project, I was used a data set containing real messages that were sent during disaster events. 

I was created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.

It include a web app where an emergency worker can input a new message and get classification results in several categories. 

The web app will also display visualizations of the data

## How to run <a name="run"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        
   `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files in the repository <a name="constructure"></a>

```
        |   README.md
        |
        +---app
        |   |   run.py
        |   |
        |   \---templates
        |           go.html
        |           master.html
        |
        +---data
        |       clf_model.pkl
        |       DisasterResponse.db
        |       disaster_categories.csv
        |       disaster_messages.csv
        |       process_data.py
        |
        \---models
                classifier.pkl
                train_classifier.py
```