import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from recipe_recommendation import preprocess_data, RecipeDataset, RecipeBERT, generate_embeddings
from sklearn.metrics import accuracy_score

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
file_path = "recipe_enhanced_v2.csv"
df = preprocess_data(pd.read_csv(file_path))  # Now df has 'label' column

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize Model
model = RecipeBERT()

# Load fine-tuned model weights
model.load_state_dict(torch.load("fine_tuned_model.pt", map_location=device))
model.eval()
model.to(device)

MAX_LEN = 128

# Prepare Dataset and DataLoader for Accuracy Check
val_texts = df['Cleaned_Ingredients'].tolist()
val_labels = df['label'].tolist()

val_dataset = RecipeDataset(val_texts, val_labels, tokenizer, MAX_LEN)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

def calculate_accuracy(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs).squeeze()
            preds = (preds > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

# Calculate Validation Accuracy
accuracy = calculate_accuracy(model, val_loader, device)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Generate Embeddings using the fine-tuned model
recipe_embeddings = generate_embeddings(model, df["Cleaned_Ingredients"].tolist(), tokenizer, max_len=128, device=device)

# Save Embeddings
np.save("recipe_embeddings.npy", recipe_embeddings)
print(f"Embeddings saved! Total embeddings: {recipe_embeddings.shape[0]}")
print(f"Sampled dataset size: {len(df)} rows")
