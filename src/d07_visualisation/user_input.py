import sys
sys.path.insert(0, "/home/apprenant/simplon_projects/foodflix_engine")

from sklearn.metrics.pairwise import linear_kernel
from src.d01_load_data.load_data import listings
from src.d04_modelling.train_model import tf, tfidf_matrix


def get_similar_products():
    user_input = input("Enter product name:")
    print("product name is: " + user_input)

    user_matrix = tf.transform([user_input])
    cosine_similarities = linear_kernel(user_matrix, tfidf_matrix)
    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[0][i], listings['id'][i]) for i in similar_indices]
    results = similar_items
    for i in range(10):
        print(listings.iloc[results[i][1]])