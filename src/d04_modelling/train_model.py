from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from d01_load_data.load_data import listings, STOPWORDSFR

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = STOPWORDSFR)
tfidf_matrix = tf.fit_transform(listings['content'])