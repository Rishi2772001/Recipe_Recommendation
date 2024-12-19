import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from sklearn.metrics.pairwise import cosine_similarity
import ast
import time
import os

# Step 1: Preprocess Data
def extract_cuisine(tags):
    global_cuisines = [
        'mexican', 'american', 'italian', 'indian', 'chinese', 'japanese',
        'french', 'north-american', 'canadian', 'mediterranean', 'asian'
    ]
    try:
        tag_list = ast.literal_eval(tags.lower())
        for tag in tag_list:
            if tag in global_cuisines:
                return tag
    except:
        pass
    return "other"

def preprocess_data(df):
    # Extract cuisine from tags
    df['cuisine'] = df['tags'].apply(extract_cuisine)
    
    # Clean ingredients column
    df['Cleaned_Ingredients'] = df['ingredients'].fillna('').apply(
        lambda x: ' '.join(x.lower().split(','))
    )
    
    # Stratified + Random Sampling
    stratified_sample = df.groupby('cuisine', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 200), random_state=42)
    )
    random_sample = df.sample(min(2000, len(df)), random_state=42)
    df_sampled = pd.concat([stratified_sample, random_sample]).drop_duplicates()

    # Limit final dataset size
    final_sample_size = min(4000, len(df_sampled))
    df_sampled = df_sampled.sample(final_sample_size, random_state=42)
    
    # Create label column
    df_sampled['label'] = df_sampled['cuisine'].factorize()[0]

    return df_sampled

# Training Loop with Progress Tracking
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=3):
    for epoch in range(epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Log validation progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"Validation Batch {batch_idx + 1}/{len(val_loader)}: Loss = {loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

# Step 2: Prepare Dataset for BERT
class RecipeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Step 3: Fine-tune BERT Model
class RecipeBERT(torch.nn.Module):
    def __init__(self):
        super(RecipeBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # outputs[1] = pooled_output (CLS)
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)

# Step 4: Generate Embeddings for Recipes
def generate_embeddings(model, texts, tokenizer, max_len, device):
    model.eval()
    embeddings = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processed {i}/{len(texts)} recipes")

        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # Directly use the BERT model for embeddings
            outputs = model.bert(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                return_dict=True
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)

    return np.vstack(embeddings)

# Step 5: Recommend Recipes Using Cosine Similarity
def recommend_recipes(input_ingredients, df, recipe_embeddings, model, tokenizer, device, MAX_LEN, top_n=5):
    input_embedding = generate_embeddings(model, [input_ingredients], tokenizer, MAX_LEN, device)
    scores = cosine_similarity(input_embedding, recipe_embeddings)
    top_indices = np.argsort(scores[0])[::-1][:top_n]
    return df.iloc[top_indices][['name', 'cuisine', 'ingredients', 'steps', 'image_url']]


if __name__ == "__main__":
    # Load dataset
    file_path = 'recipe_enhanced_v2.csv'
    df = preprocess_data(pd.read_csv(file_path))

    # Tokenizer
    MAX_LEN = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Cleaned_Ingredients'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42
    )

    train_dataset = RecipeDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = RecipeDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Detect device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, optimizer, and loss function
    model = RecipeBERT()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=3)

    # Save the trained model weights
    torch.save(model.state_dict(), "fine_tuned_model.pt")
    print("Model weights have been saved to fine_tuned_model.pt")
