# COVID19 Explorations

Various exploratory analyses on Scholarly Communications relating to the COVID-19 virus and pandemic.

## Keyword searches
We've found a few keywords relating to the COVID-19 outbreak. Research publishers can use these keywords in 2 ways:
1. Use them to find related articles published on your website. 
  - You can then make those articles free-to-read if they are not free already. 
  - You might also wish to make them into a special collection so that it is easy for people to find.
2. Find related articles in peer-review and then prioritise the peer-review of those articles. 
  - You might also consider recommending preprinting to authors of articles relating to the COVID-19 outbreak.

There is a 'keyword extraction' notebook for exporing for keywords and phrases relating to coronavirus. However, the regex-based methods in tools.py are perhaps more effective as search tools for finding related content.

## Text visualisation
Text visualisations can help to get an overall picture of a corpus of documents. This can show us a lot of things depending on how we build our visualisation. In this case, there are several obvious clusters and these correspond to scientific fields. By comparing papers from the CORD-19 dataset to the rest of pubmed, we can see which fields of pubmed are most relevant to the COVID-19 outbreak.

Text data is inherently noisy and analysis of it is not a precision science. When we make data visualisations like this, we lose a lot of detail in the process of compressing our data into a 2-dimensional representation, so it's important to understand that there are limitations to what we can learn from a visualisation like this.

__Note:__ if you want to do text vectorisation with SciSpaCy, you might need to install it in a separate virtual environment. Installing alongside SpaCy (in the environment for this repo) seems to cause problems.

``` bash
conda create scispacy
conda activate scispacy
pip install scispacy pandas numpy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_sm-0.2.3.tar.gz
```

## Text classification
A classifier is something which we can use to tell 2 (or more) things apart. In this case, we build a simple classifier to distinguish between the CORD-19 dataset and PubMed. This has a few uses. Much like keyword analysis above:
- It can tell us if a paper we have published is potentially relevant to the COVID-19 outbreak. Importantly, it can even do this if the paper does not contain any obvious keywords. Publishers have generally agreed to make relevant research free-to-read, so this classifier can help with the process of identifying that research and making it available (or even promoting to make it more visible)
- It can also tell us if a paper we have in peer-review is relevant to the outbreak and that can help us to prioritise peer-review.
- On the other hand, it can also tell us when a paper is much less likely to have anything to do with the outbreak and this can help to put a lower priority on that content and focus resources on the more urgent work. 

Note that, the Type 1 and Type 2 errors from this classifier are quite interesting. Sometimes these are genuine errors - and we should be mindful that no classifier will be perfect. However, other times they show articles in CORD-19 which are not highly relevant to the outbreak and other times they show articles in PubMed which ARE relevant, but were not included in the CORD-19 dataset. 


