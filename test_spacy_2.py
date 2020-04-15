

import pandas as pd
import numpy as np
import spacy

spacy.prefer_gpu()
output_dir = 'models/spacy_textcategorizer'
nlp = spacy.load(output_dir)

from sklearn.metrics import confusion_matrix
from tools import pre_s, add_check_cols
import re
import os

print()
print('test with SAGE peer-review data')

data_path = os.path.abspath('data/2020_subs_to_date_inc_lifecycle_pre.csv')
if os.path.exists(data_path):
    test = pd.read_csv(data_path, dtype = str)
    test = test.drop_duplicates('dw_submission_key', keep = 'first')
    # test['abstract_pre'] = test['abstract_text'].map(lambda x:pre_s(str(x).lower()))
    print('Dataframe shape: ', test.shape)
    # preprocess
    test['tiabs'] = test['submission_title'] +'. '+test['abstract_pre']
    test['tiabs'] = test['tiabs'].map(lambda x: str(x).lower())
    test = add_check_cols(test)

    out = []
    for i,row in test.iterrows():
        text = str(row['tiabs'])
        abstract = str(row['abstract_pre'])
        if (len(text.split())>=20) and (len(text.split())<500):
            doc = nlp(text)
            out.append((doc.cats['POSITIVE'],doc.cats['NEGATIVE']))
        elif 'no abstract' in abstract or 'abstract not req' in abstract:
            out.append(('********* Abstract missing? *********','********* Abstract missing? *********'))
        else:
            out.append(('********* Input too short/long *********','********* Input too short/long *********'))


    test['prob_cov'] = [x[0] for x in out]
    test['prob_pubmed'] = [x[1] for x in out]
    test['pred'] = [1 if type(x[0])==float and x[0]>=0.5 else 0 for x in out]
    print('Correlation: ')
    print(test[['strong_kw_match','pred']].corr())
    print('Confusion:')
    print(confusion_matrix(test['strong_kw_match'].values, test['pred'].values))

    out_cols = [
        'submission_id_original', 'submission_id', 'submission_id_latest', 'submission_date',
    'withdrawn_date',    'journal_name', 
    'preprint_information', 'revision_number',
    'submission_title', 'abstract_text', 'abstract_pre', 'tiabs',
    'covid-19', 'coronavirus', 'mers_sars', 'flu', 'strong_kw_match',
    'pandemic', 'vaccine', 'zoonosis', 'virus', 'wuhan', 'weak_kw_match',
    'prob_cov', 'prob_pubmed', 'pred'
    ]
    test = test[out_cols]
    # sort cols
    test = test.sort_values(['covid-19', 'coronavirus', 'mers_sars', 'flu','pandemic'], ascending = False)

    test.to_csv('output/test_spacy_peerreview_out.csv', encoding = 'utf-8-sig',index=False)
    print("Saved test to", 'output/test_spacy_peerreview_out.csv')
else:
    print('unable to find peer-review data. Ending script.')