

import pandas as pd
import numpy as np
import spacy

spacy.prefer_gpu()
output_dir = 'models/spacy_textcategorizer'
nlp = spacy.load(output_dir)

from sklearn.metrics import confusion_matrix
from tools import pre_s, add_check_cols
import re

print('Here is a test doc with the word \'coronavirus\' removed. The classifier should still be able to recognise that is it relevant by giving it a high score (close to 1)')
test_doc = """Preparation and characterization of SARS in-house reference antiserum. A reference antiserum for SARS is in urgent need in the development of SARS vaccine and other serological test of SARS research. Convalescent serum was collected from clinical confirmed patient. ELISA, Western-blotting and neutralization assay detected specific antibody against SARS. This antiserum was prepared as in-house reference antiserum, freeze-dried and sealed in ampoules. The potency of this reference antiserum is defined to be 52.7U after extensive calibration. Further, collaborative studies for the evaluation of this serum are needed in order to satisfy the requirements for international reference antiserum."""
print('EXAMPLE DOCUMENT:',test_doc)

doc = nlp(test_doc)
prob = doc.cats['POSITIVE']

print('Probability = ',prob)
print()


print('Test with test data (mix of cord-19 with pubmed)')
test = pd.read_csv('data/test.csv', dtype = str)

# preprocess
test['tiabs'] = test['tiabs'].map(lambda x: str(x).lower())

test = test.rename(columns={'covid':'cord-19'})

# keyword checks
test = add_check_cols(test)


out = []
for i,row in test.iterrows():
    doc = nlp(row['tiabs'])
    out.append((doc.cats['POSITIVE'],doc.cats['NEGATIVE']))
test['prob_cov'] = [x[0] for x in out]
test['prob_pubmed'] = [x[1] for x in out]
test['pred'] = [1 if x[0]>=0.5 else 0 for x in out]
test['cord-19'] = test['cord-19'].map(lambda x: int(x))


print('Correlation: ')
print(test[['cord-19','pred']].corr())
print('Confusion cord-19/pred:')
print(confusion_matrix(test['cord-19'].values, test['pred'].values))
print('Confusion cord-19/strong_kws:')
print(confusion_matrix(test['cord-19'].values, test['strong_kw_match'].values))
print('Confusion pred/strong_kws:')
print(confusion_matrix(test['pred'].values, test['strong_kw_match'].values))


out_cols = [
   'venue',  'articledate', 'doi', 'pmid', 'pii', 'pmc', 
 'title', 'abstract', 'year','abstract_pre', 'tiabs', 'lang', 'delta',
 'cord-19', 'covid-19', 'coronavirus', 'mers_sars', 'flu', 'strong_kw_match',
    'pandemic', 'vaccine', 'zoonosis', 'virus', 'wuhan', 'weak_kw_match',
    'prob_cov', 'prob_pubmed', 'pred'
    ]
test = test[out_cols]
# sort cols to make it easy to inspect manually
test = test.sort_values(['covid-19', 'coronavirus', 'mers_sars', 'flu','pandemic'], ascending = False)

test.to_csv('output/test_spacy_out.csv', encoding = 'utf-8-sig',index=False)
print("Saved test to", 'output/test_spacy_out.csv')

