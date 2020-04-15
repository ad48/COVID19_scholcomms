import pandas as pd
import numpy as np
import spacy
import datetime


if __name__ =='__main__':

    nlp = spacy.load("en_core_sci_lg")

    print(datetime.datetime.now(),' -------------- Loading dataframes')
    train_scispacy =pd.read_csv('data/train_scispacy.csv', dtype=str)
    test_scispacy = pd.read_csv('data/test_scispacy.csv', dtype=str)
    train =pd.read_csv('data/train.csv', dtype=str)
    test = pd.read_csv('data/test.csv', dtype=str)
    dev = pd.read_csv('data/dev.csv', dtype=str)
    all_ = pd.read_csv('data/all_s2_pubmed.csv', dtype=str)
    print(datetime.datetime.now(),' -------------- Dataframes loaded')

    dataframes = {  
                    'dev_scispacy_vecs': dev,
                    'test_scispacy_vecs': test,
                    'test_scispacy_vecs_bal': test_scispacy,
                    'train_scispacy_vecs_bal': train_scispacy,
                    'train_scispacy_vecs': train,
                    'all_scispacy_vecs': all_,
                 }


    for dataframe in dataframes:
        df = dataframes[dataframe]
        docs = df['tiabs'].values
        print(datetime.datetime.now(),' -------------- Building array for', dataframe)
        docs = list(nlp.pipe(docs))
        arr = np.array([doc.vector for doc in docs])
        print(datetime.datetime.now(),' -------------- ',arr.shape, ' array created.')
        np.save('data/{}.npy'.format(dataframe),arr)
        print(datetime.datetime.now(),' -------------- ', dataframe, ' vectorised and written to file.')
