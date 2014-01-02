__author__ = 'nastra'

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os


english_stemmer = SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def distance_raw(v1, v2):
    delta = v1 - v2
    return np.linalg.norm(delta.toarray())

def distance_normalized(v1, v2):
    v1_normalized = v1 / np.linalg.norm(v1.toarray())
    v2_normalized = v2 / np.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return np.linalg.norm(delta.toarray())

def load_data_from_dir(directory, delimiter):
    files = [open(os.path.join(directory, f)).read() for f in os.listdir(directory)]
    out = []
    for f in files:
        out.extend(f.split(delimiter))
    return out

def get_similar_posts(X, post, posts):
    import sys
    shortest_dist = sys.maxint
    num_samples, num_features = X.shape
    post_vectorized = vectorizer.transform([post])
    best_post = None
    best_post_index = None

    for i in range(0, num_samples):
        current_post = posts[i]
        if current_post == post:
            continue
        curr_post_vectorized = X.getrow(i)
        dist = distance_normalized(curr_post_vectorized, post_vectorized)
        print "Post %i: '%s' with distance= %.2f" % (i, current_post, dist)
        if dist < shortest_dist:
            shortest_dist = dist
            best_post_index = i
            best_post = current_post

    if best_post_index is not None:
        return X.getrow(best_post_index), best_post, shortest_dist
    return None, None, None




vectorizer = StemmedCountVectorizer(min_df=1)
posts = load_data_from_dir("Building_ML_Systems_with_Python/chapter_03_Codes/data/toy", "\n")

X_train = vectorizer.fit_transform(posts)
post = "support vector machine"
post_vec, found_post, distance = get_similar_posts(X_train, post, posts)

print "\n"
print "The most similar post to '%s' is: '%s' with distance= %.2f" % (post, found_post, distance)


# TODO: extract topics from stackoverflow and measure similarity to a given question


