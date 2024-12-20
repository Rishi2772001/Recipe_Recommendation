
# Global Recipe Recommendation System

This project provides a global recipe recommendation system using a fine-tuned BERT model. It takes in user-input ingredients and desired cuisine to return a list of recommended recipes along with their ingredients and steps. The system leverages the power of NLP-based embeddings to find semantically similar recipes quickly and accurately.

## Features

-   **Data Preprocessing:**  
    Automatically extracts cuisines from recipe tags, cleans ingredient lists, and samples the dataset for balanced training.
    
-   **Model Fine-tuning:**  
    A  `BertModel`  is fine-tuned on the preprocessed dataset to learn meaningful embeddings for recipes, enabling similarity-based recommendations.
    
-   **Precomputed Embeddings:**  
    Generates and saves embeddings for all recipes to speed up recommendation queries.
    
-   **Interactive UI with Streamlit:**  
    A user-friendly web interface allows users to input ingredients and select a cuisine to retrieve recommended recipes with steps and images.
    
       

## Setup and Installation

1.  **Clone the Repository:**
    
    bash
    
    `git clone <repository-url>
    cd <repository-directory>` 
    
2.  **Set Up Python Environment:**  It’s recommended to use a virtual environment (e.g.,  `pyenv`  or  `venv`):
    
    bash
    
    `python3 -m venv venv
    source venv/bin/activate` 
    
3.  **Install Dependencies:**
    
    bash
    
    `pip install -r requirements.txt` 
    
    Ensure that  `torch`,  `transformers`,  `pandas`,  `streamlit`,  `scikit-learn`, and  `numpy`  are installed. If not, install them individually:
    
    bash
    
    `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers pandas scikit-learn numpy streamlit` 
    
4.  **Download the Dataset:**  You can find the enhanced recipes CSV file [here](https://www.kaggle.com/datasets/behnamrahdari/foodcom-enhanced-recipes-with-images).

    

## Running the Project

### 1. Train the Model

Run the training script to preprocess data and fine-tune the BERT model:

bash

`python3 recipe_recommendation.py` 

This will:

-   Load and preprocess the data.
-   Train BERT on the processed dataset.
-   Save the fine-tuned model weights to  `fine_tuned_model.pt`.

### 2. Precompute Embeddings

Next, generate embeddings for all recipes using the fine-tuned model:

bash

`python3 precompute_embeddings.py` 

This will:

-   Load the fine-tuned model.
-   Compute embeddings for each recipe.
-   Save embeddings to  `recipe_embeddings.npy`.

### 3. Run the Streamlit App

Finally, launch the Streamlit application:

bash

`streamlit run app.py` 

Open the provided local URL in your browser. You can then:

-   Enter ingredients (comma-separated).
-   Choose a desired cuisine.
-   Click "Find Recipes" to see recommendations.

## How It Works

1.  **Preprocessing & Training:**  
    The model is fine-tuned to understand the relationship between ingredients and cuisines, providing a way to represent recipes as embeddings.
    
2.  **Embedding Computation:**  
    After training, each recipe is transformed into a vector (embedding) that captures its semantic characteristics.
    
3.  **Recommendations:**  
    When the user inputs ingredients and selects a cuisine, the system generates an embedding for those ingredients and finds the most similar recipe embeddings via cosine similarity.
    
4.  **Display Results:**  
    The top recommended recipes are displayed with their ingredients and steps in a user-friendly manner.

## Troubleshooting

-   **Missing Dependencies:**  
    Ensure you have the correct Python environment activated and that all required libraries are installed.
    
-   **Torch Not Found:**  
    Make sure you run  `streamlit run app.py`  from the same environment in which  `torch`  is installed.
    
-   **No Recipes Found:**  
    Try different ingredients or ensure that the dataset contains recipes for the selected cuisine.
    

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, fixes, or new features.

