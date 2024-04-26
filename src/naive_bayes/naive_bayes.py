import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


data = pd.read_csv('../../data/Mental-Health-Twitter.csv')


def preprocess_tweet(tweet):
    tweet = tweet.lower() #lowercase
    tweet = re.sub(r'http\S+', '', tweet) # remove URLs
    # handle emojis (TO DO: this could be more advanced -- right now evaluates a limited/simplistic set of emojis)
    emojis = re.findall('[:;=8xX][\-]?[)D\(pP/\\:\}\{@\|]', tweet)
    positive_emojis = sum([1 for e in emojis if e in [':)', ':D', ';)', ':-)']])
    negative_emojis = sum([1 for e in emojis if e in [':(', 'D:', ':-(']])
    tweet = re.sub(r'[:;=8xX][\-]?[)D\(pP/\\:\}\{@\|]', '', tweet)
    # remove non-alphabetic characters
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    # remove stopwords, source: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokenize(tweet) if word not in stop_words]
    return ' '.join(words), positive_emojis, negative_emojis

data['Processed'], data['PosEmo'], data['NegEmo'] = zip(*data['post_text'].apply(preprocess_tweet))

# feature engineering w/ TF-IDF (from scikit documentation) 
tfidf = TfidfVectorizer() 
X_text = tfidf.fit_transform(data['Processed'])  # processed tweets = corpus, each processed tweet = row, each cell = single term's TF-IDF score 
X_emo = np.vstack((data['PosEmo'], data['NegEmo'])).T #include emojis
X = np.hstack((X_text.toarray(), X_emo)) # x = feature matrix (to be used in testing)

y = data['label']

# TO DO: make this k-fold cross validation instead (to avoid duplicates / overfitting)
num_splits = 5 # multiple train/test splits
accuracy, precision, recall, f1 = [], [], [], []

for _ in range(num_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # evaluate model
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1.append(f1_score)

# model's performance metrics
print(f"Average Accuracy: {np.mean(accuracy):.2f}")
print(f"Average Precision: {np.mean(precision):.2f}")
print(f"Average Recall: {np.mean(recall):.2f}")
print(f"Average F1 Score: {np.mean(f1):.2f}")