import streamlit as st
import sys
sys.path.insert(0, "/home/apprenant/simplon_projects/foodflix_engine/")

from d01_load_data import load_data
from d04_modelling import train_model
from d07_visualisation.user_input import get_similar_products

get_similar_products()

