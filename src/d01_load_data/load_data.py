# Importing the libraries
import pandas as pd
from IPython.display import Image, HTML
import matplotlib.pyplot as plt



# Importing the dataset
listings = pd.read_csv('/home/apprenant/simplon_projects/foodflix_engine/data/raw/listings.csv')
listings['id'] = listings.index
listings = listings[['id', 'product_name', 'brands', 'ingredients_text']]
listings.head(10)

#FillNa
listings.product_name.fillna('Null', inplace = True)
listings.brands.fillna('Null', inplace = True)
listings.ingredients_text.fillna('Null', inplace = True)

listings['product_name'] = listings['product_name'].astype('str')
listings['brands'] = listings['brands'].astype('str')
listings['ingredients_text'] = listings['ingredients_text'].astype('str')

product_name_corpus = ' '.join(listings['product_name'])
brands_corpus = ' '.join(listings['brands'])
ingredients_corpus = ' '.join(listings['ingredients_text'])


listings['content'] = listings[['product_name', 'brands', 'ingredients_text']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)




STOPWORDSFR = {'le', 'la', 'de', 'du', 'au', 'les', 'aux', 'null'}