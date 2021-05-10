import sys
sys.path.insert(0, "/home/apprenant/simplon_projects/foodflix_engine")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.d01_load_data.load_data import listings, STOPWORDSFR
#from transformers import CamembertModel, CamembertTokenizer

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = STOPWORDSFR)
tfidf_matrix = tf.fit_transform(listings['content'])



# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
#tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
#camembert = CamembertModel.from_pretrained("camembert-base")

#camembert.eval()  # disable dropout (or leave in train mode to finetune)


#from sentence_transformers import SentenceTransformer
#tf = SentenceTransformer('distiluse-base-multilingual-cased-v1')


#tfidf_matrix = tf.encode(listings['content'])

