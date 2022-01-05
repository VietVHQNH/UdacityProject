import os
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Args:
        messages_filepath   --> str: path to message CSV file
        categories_filepath --> str: path to categories CSV file
    Return:
        Dataframe which is combied data from messages data and categories data
    """
    message_df = pd.read_csv(messages_filepath)
    categorie_df = pd.read_csv(categories_filepath)
    return pd.merge(message_df, categorie_df, on='id')


def clean_data(df):
    """
    Args:
        df  --> pandas dataframe 
    Return:
        Cleaned dataframe
    """    
    # Create the copy dataframe
    dump_df = df.copy()
    # Splits the categories column into separate
    categories_df = dump_df['categories'].str.split(";",expand = True)
    # Get new column names from categories values
    columns = [x.split("-")[0] for x in categories_df.iloc[0].values]
    # Change the values in categores dataframe to binary values
    categories_df = categories_df.applymap(lambda x:0 if x.split("-")[-1]=='0' else 1)
    # Change comlumns name from 1,..,36 to respective values
    categories_df.columns = columns
    # Return concatenate the original dataframe with the new `categories` dataframe and drop the duplicates
    output = pd.concat([df.drop(['categories'], axis = 1),categories_df], axis = 1)
    return output.drop_duplicates()



def save_data(df, database_filepath):
    """
    Args:
        df  --> pandas dataframe 
        database_filename --> data file name
    Return:
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = os.path.basename(database_filepath).split(".")[0]
    df.to_sql(table_name, engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()