import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
sys.path.insert(0, "/home/apprenant/simplon_projects/foodflix_engine/")

DATA_PATH = '/home/apprenant/simplon_projects/foodflix_engine/data/raw/listings.csv'
@st.cache(allow_output_mutation=True)
def load_data(path):
    # Importing the dataset
    listings = pd.read_csv(path)
    listings['id'] = listings.index
    listings = listings[['id', 'product_name', 'brands', 'ingredients_text', 'nutrition_grade_fr']]
    #FillNa
    listings.product_name.fillna('Null', inplace = True)
    listings.brands.fillna('Null', inplace = True)
    listings.ingredients_text.fillna('Null', inplace = True)

    listings['product_name'] = listings['product_name'].astype('str')
    listings['brands'] = listings['brands'].astype('str')
    listings['ingredients_text'] = listings['ingredients_text'].astype('str')
    #product_name_corpus = ' '.join(listings['product_name'])
    #brands_corpus = ' '.join(listings['brands'])
    #ingredients_corpus = ' '.join(listings['ingredients_text'])
    listings['content'] = listings[['product_name', 'brands', 'ingredients_text']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
    return listings

@st.cache(allow_output_mutation=True)
def create_model(data, model):
    if model == 'TfidfVectorizer':
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = STOPWORDSFR)
    elif model == 'CountVectorizer':
        tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = STOPWORDSFR)
    else:
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = STOPWORDSFR)     
    tfidf_matrix = tf.fit_transform(data)
    return tf, tfidf_matrix

def get_similar_products(input, slider_input=5):
    user_matrix = tf.transform([input])
    cosine_similarities = linear_kernel(user_matrix, tfidf_matrix)
    similar_indices = cosine_similarities[0].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[0][i], listings['id'][i]) for i in similar_indices]
    results = []
    for i in range(slider_input):
        results.append(listings.iloc[similar_items[i][1]])
    return results



if __name__ == '__main__':
    st.title('Foodflix')
    st.text('Source des données : OpenFoodFacts.org')

    slider_input = st.sidebar.slider('Nombre de résultats', min_value=5, max_value=20)
    model_input = st.sidebar.radio('Choix du modèle NLP', ['TfidfVectorizer','CountVectorizer','BERT'])

    listings = load_data(DATA_PATH)
    CONTENT = listings['content']
    STOPWORDSFR = {'le', 'la', 'de', 'du', 'au', 'les', 'aux', 'null'}
    tf, tfidf_matrix = create_model(CONTENT, model_input)
    user_input = st.text_input("Chercher un produit", value='', max_chars=60, key=None, type='default', help=None)
    if user_input:
        results = get_similar_products(user_input, slider_input)
        for item in results:
            st.title('Marque: {}'.format(item['brands']))
            st.write('Nom du produit: {}'.format(item['product_name']))
            st.write('Liste des ingrédients: {}'.format(item['ingredients_text']))
 
            if item['nutrition_grade_fr'] == 'a':
                st.image('assets/a.png')
            if item['nutrition_grade_fr'] == 'b':
                st.image('assets/b.png')
            if item['nutrition_grade_fr'] == 'c':
                st.image('assets/c.png')
            if item['nutrition_grade_fr'] == 'd':
                st.image('assets/d.png')
            if item['nutrition_grade_fr'] == 'e':
                st.image('assets/e.png')

             
            st.write('---')


