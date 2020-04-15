from nltk import sent_tokenize
import re
import pandas as pd

def drop_copyright_statements(abstract):
    chars = ['©','(c)','copyright','all rights reserved', 'elsevier']
    paras = abstract.split('\n')
    paras = [x for x in paras if not any([char in x.lower() for char in chars])]
    return ' '.join(paras).strip()

def regex_strip(sentence, term):
    dropchars = ['/','&','and ','of ','the ']
    for dropchar in dropchars:
        sentence = sentence.strip()
        regex = "\A{}[\s]?".format(dropchar)
        sentence = re.sub(regex, '', sentence, re.IGNORECASE)
    
    # anywhere with a colon - sometimes sent_tokenize doesn't split on a '.'
    regex = "[\.\")”] {}[s]?[:]?".format(term)
    sentence = re.sub(regex, '. ', sentence, re.IGNORECASE)
    
    # at the start of a sentence with/without a colon
    regex = "\A{}[s]?[\s]?[:]?".format(term)
    sentence = re.sub(regex, '', sentence, re.IGNORECASE)
    
    return sentence


def remove_subheadings(sentence):
    """
    This is very hacky. 
    The subheadings are such an obvious problem that I'll bet that someone has written a much better function for this somewhere...
    """
    
    drop_subheadings = ['Abstract','Background','Patient','Hypothesis','Hypotheses',
                        'Publisher Summary', 'Comparison','Subject','Key','Cohort',
                    'Search','Simple','Main',"Author's","Authors'", "Editor's","Editors'",
                    'Lay','General',
                    'Study Design','Design and Methods','Methods and Design',
                    'Materials and methods','Method','Aim','Case',
                        'Introduction', 'Purpose', 'Material','Study','Significance',
                        'Design',
                        'Summary','Objective','Result','Conclusion','Setting','Discussion',
                       'Areas covered','Expert opinion','Data Synthesis','Data Source',
                        'Study Selection and Data Extraction','Measurements and Results','Main Methods',
                       'Case Report','Methodology','Project Outline',]
    for term in drop_subheadings:
        term=term.lower() # re.IGNORECASE didn't catch them 
        sentence = regex_strip(sentence, term)
    return sentence


def simple_remove_subheadings(sentence):
    """
    Where there is a colon present, this ought to work. 
    Still cases where colons are missing or where subheading is very long
    e.g. "Abstract blah blah", "Study Design and Key Methods". Stuff like that.
    Also cases where sent_tokenize doesn't split in the right place.
    """
    if ':' in sentence:
        loc = sentence.find(':')
        before = sentence[:loc]
        len_before = len(before.split())
        if len_before<=3:
            sentence = sentence[loc:]
    return sentence


def pre_s(abstract):
    """
    Rough function - needs work
    Take an abstract and look for obvious copyright statements. Remove them.
    Sometimes weird statement with publishers'/authors' email address(?) Remove that too.
    """
    abstract = drop_copyright_statements(abstract)
    sents = sent_tokenize(abstract)
    new_sents = []
    for i,sent in enumerate(sents):
        sent = simple_remove_subheadings(sent)
        if i==len(sents) and '@' in sent:
            pass
        elif "electronic supplementary material" in sent:
            pass
        else:
            sent = remove_subheadings(sent)
            new_sents.append(sent)
    new_sents = ' '.join(new_sents)
    return new_sents


## DATAFRAME PREPROCESSING



def check_kws(text, kw_dict):
    text = str(text)
    out = {}
    for x in kw_dict:
        out[x] = 0
        regex = kw_dict[x]
        if re.search(regex,text):
            out[x] = 1
    return out

def add_check_cols(df):
    print('adding keyword checks to dataframe', df.shape)
    df = df.reset_index() # required or concatenation gives odd behaviour
    definites = {
    'mers_sars': r'\b(respiratory syndrome|mers|sars)\b',
    'covid-19': r'\b(covid[-]?(19)?|sars[-\s]ncov[-\s]?[2]?|ncov[-\s]?2019|2019[-\s]?ncov)\b',
    'coronavirus': r'corona[-\s]?vir(us|idae)\b',
    'flu': r'\b(flu|influenza|h1n1|h5n1)\b',
    }

    maybes = {
        'pandemic': r'\b(epi|pan)demic[s]?\b',
        'vaccine': r'\bvaccin(e[s]?|ate[ds]?|ation[s]?)\b',
        'zoonosis': r'\bzoon(osis|otic)\b',
        'virus': r'\b(vir(us|uses|al|ology|ological)|anti[-]?bod(y|ies))\b',
        'wuhan': r'\bwuhan\b',
    }

    kw_dict = definites
    text_data = df['tiabs'].tolist()
    columns = list(kw_dict.keys()) + ['strong_kw_match']
    rows = []
    for text in text_data:
        row = check_kws(text,kw_dict)
        any_ = int((sum([row[x] for x in row])>0))
        row['strong_kw_match'] = any_
        rows.append(row)
    check_df = pd.DataFrame(rows, columns = columns)
    df = pd.concat([df, check_df], axis =1)
    kw_dict = maybes
    columns = list(kw_dict.keys()) + ['weak_kw_match']
    rows = []
    for text in text_data:
        row = check_kws(text,kw_dict)
        any_ = int((sum([row[x] for x in row])>0))
        row['weak_kw_match'] = any_
        rows.append(row)
    check_df = pd.DataFrame(rows, columns = columns)
    df = pd.concat([df, check_df], axis =1)
    return df