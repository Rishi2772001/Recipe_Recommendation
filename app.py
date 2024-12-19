import streamlit as st
import pandas as pd
import ast
import numpy as np
import torch
from transformers import BertTokenizer
from recipe_recommendation import recommend_recipes, preprocess_data, RecipeBERT

@st.cache_data
def load_data():
    file_path = 'recipe_enhanced_v2.csv'
    df = pd.read_csv(file_path)
    return preprocess_data(df)

# Load and preprocess data
df = load_data()
unique_cuisines = df['cuisine'].unique()

# Load precomputed embeddings
recipe_embeddings = np.load("recipe_embeddings.npy")

# Load model and tokenizer
MAX_LEN = 128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = RecipeBERT()
model.eval()
model.to(device)

# Streamlit App Title
st.title("Global Recipe Recommendation System 🍲")

# User Input Section
st.write("Find recipes based on ingredients and selected cuisine!")
ingredients_input = st.text_area("Enter ingredients (comma-separated):")
selected_cuisine = st.selectbox("Select a cuisine:", unique_cuisines)

# Button to Generate Recommendations
if st.button("Find Recipes"):
    if ingredients_input and selected_cuisine:
        try:
            # Filter recipes by cuisine
            filtered_df = df[df['cuisine'] == selected_cuisine].reset_index(drop=True)
            if not filtered_df.empty:
                # Filter embeddings for selected cuisine
                indices = filtered_df.index.tolist()
                filtered_embeddings = recipe_embeddings[indices]

                # Get recommendations
                recommendations = recommend_recipes(
                    ingredients_input, filtered_df, filtered_embeddings, 
                    model, tokenizer, device, MAX_LEN
                )
                if not recommendations.empty:
                    st.write(f"### Recommended {selected_cuisine.capitalize()} Recipes:")
                    for _, row in recommendations.iterrows():
                        st.subheader(row['name'])
                        st.write("**Cuisine:**", row['cuisine'].capitalize())

                        # Display ingredients
                        ingredients = ', '.join(ast.literal_eval(row['ingredients']))
                        st.write("**Ingredients:**", ingredients)

                        # Display steps as an unordered list
                        st.write("**Steps:**")
                        steps = ast.literal_eval(row['steps'])
                        for step in steps:
                            st.markdown(f"- {step}")

                else:
                    st.warning(f"No recipes found for {selected_cuisine} with the given ingredients.")
            else:
                st.warning(f"No recipes available for the selected cuisine.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter ingredients and select a cuisine.")
