'''
Reference:
    https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
'''


# TfidfVectorizer
# CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# set of documents
train = ['The sky is blue.', 'The sun is bright.']
test = ['The sun in the sky is bright', 'We can see the shining sun, the bright sun.']
# instantiate the vectorizer object
countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
# convert th documents into a matrix
count_wm = countvectorizer.fit_transform(train)
tfidf_wm = tfidfvectorizer.fit_transform(train)
# retrieve the terms found in the corpora
# if we take same parameters on both Classes(CountVectorizer and TfidfVectorizer) , it will give same output of get_feature_names() methods)
# count_tokens = tfidfvectorizer.get_feature_names() # no difference
count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_countvect = pd.DataFrame(data=count_wm.toarray(), index=['Doc1', 'Doc2'], columns=count_tokens)
df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), index=['Doc1', 'Doc2'], columns=tfidf_tokens)
print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer with index\n")
print(df_tfidfvect)

print("\nTD-IDF Vectorizer without index\n")
print(pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens))