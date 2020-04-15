# adapted from https://spacy.io/usage/examples#textcat
# this should run perfectly well on a CPU, but a GPU will be faster
# consider installing spacy with GPU support https://spacy.io/usage
# can't get this to work with scispacy. Pity.

#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
# import thinc.extra.datasets
import pandas as pd
# import scispacy
import spacy
from spacy.util import minibatch, compounding

from tools import pre_s





@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path),
)
def main(model=None, output_dir=None, n_iter=20, n_texts=900000, init_tok2vec=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path('models/spacy_textcategorizer')
    if not output_dir.exists():
        output_dir.mkdir()
    # use gpu if available
    spacy.prefer_gpu()
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")

    # load the dataset
    print("Loading data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )

    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                
                texts, annotations = zip(*batch)
                # print(texts, annotations)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # test the trained model
    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


        


def load_data(limit=0, split=0.8):
    """Load data from the pubmed/cord datasets."""
    
    
    ### OPTIONS
    ### 1
    # # use small balanced training set (fast, but might miss some features in PubMed)
    # train = pd.read_csv('data/train_scispacy.csv', dtype = str)
    # # shuffle
    # train = train.sample(train.shape[0], random_state = 7)
    # # take a % of training data to be used for dev set
    # dev = train.tail(int((1-split)*train.shape[0]))
    # # take remainder to be training set
    # train = train.head(int((split)*train.shape[0]))

    ### 2
    # use large real-world training dataset (slow, but potentially better results)
    train = pd.read_csv('data/train.csv', dtype = str)
    dev = pd.read_csv('data/dev.csv', dtype = str)
    # shuffle
    train = train.sample(train.shape[0],random_state = 100)
    dev = dev.sample(dev.shape[0],random_state = 100)

    # Now pass the data out    
    train = train[['covid','tiabs']].dropna()
    train_texts = train['tiabs'].values
    train_labels = train['covid'].values
    train_cats = [{"POSITIVE": bool(int(y)), "NEGATIVE": not bool(int(y))} for y in train_labels]

    dev = dev[['covid','tiabs']].dropna()
    dev_texts = dev['tiabs'].values
    dev_labels = dev['covid'].values
    dev_cats = [{"POSITIVE": bool(int(y)), "NEGATIVE": not bool(int(y))} for y in dev_labels]
    # return train and dev sets
    return (train_texts, train_cats), (dev_texts, dev_cats)


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    plac.call(main)