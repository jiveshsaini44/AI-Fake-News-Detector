import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cpu")  # Use CPU for i5 laptop
print(f"Using device: {device}")

class NewsDataset(Dataset):
    """Custom Dataset for news articles"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def load_and_prepare_data(file_path='C:\\Users\\HP\\Desktop\\7th\\fake_news_dataset.csv'):
    """Load and prepare dataset"""
    print(f"Loading data from {file_path}...")
    
    # Try to load CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle different dataset formats
    # Common formats: (text, label), (title, text, label), etc.
    
    # Detect text and label columns
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'content' in col_lower or 'article' in col_lower:
            if text_col is None:
                text_col = col
        if 'label' in col_lower or 'class' in col_lower or 'target' in col_lower:
            label_col = col
    
    if text_col is None or label_col is None:
        print("Could not automatically detect columns. Using first and last columns.")
        text_col = df.columns[0]
        label_col = df.columns[-1]
    
    print(f"Using text column: {text_col}")
    print(f"Using label column: {label_col}")
    
    # Clean text
    print("Cleaning text data...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 20]
    
    # Encode labels (0=fake, 1=real)
    unique_labels = df[label_col].unique()
    print(f"Unique labels: {unique_labels}")
    
    # Map labels to 0 and 1
    if len(unique_labels) == 2:
        # Assume first is fake, second is real (or vice versa)
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        # Try to detect which is which
        for label in unique_labels:
            if str(label).lower() in ['fake', 'false', '0', 'unreliable']:
                label_mapping[label] = 0
            elif str(label).lower() in ['real', 'true', '1', 'reliable']:
                label_mapping[label] = 1
        
        df['label'] = df[label_col].map(label_mapping)
        print(f"Label mapping: {label_mapping}")
    else:
        print("Warning: More than 2 labels detected. Using binary classification.")
        df['label'] = (df[label_col] == unique_labels[0]).astype(int)
    
    # Remove any NaN labels
    df = df.dropna(subset=['label'])
    
    print(f"Final dataset size: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df['cleaned_text'].values, df['label'].values

def train_model(train_loader, model, optimizer, scheduler, epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate_model(val_loader, model):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

def main():
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"  # Lightweight model for i5 laptop
    MAX_LENGTH = 512
    BATCH_SIZE = 8  # Small batch size for CPU
    EPOCHS = 3  # Start with 3 epochs
    LEARNING_RATE = 2e-5
    
    # File path - Update this with your dataset path
    DATA_PATH = 'fake_news_dataset.csv'
    
    print("="*50)
    print("FAKE NEWS DETECTOR - TRAINING SCRIPT")
    print("="*50)
    
    # Load data
    texts, labels = load_and_prepare_data(DATA_PATH)
    
    if texts is None:
        print("Failed to load data. Please check your dataset.")
        return
    
    # Split data
    print("\nSplitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = NewsDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    best_val_accuracy = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_model(
            train_loader, model, optimizer, scheduler, epoch
        )
        print(f"\nTrain Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        
        # Evaluate
        val_loss, val_acc, predictions, true_labels = evaluate_model(
            val_loader, model
        )
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"\n✓ New best model! Saving...")
            model.save_pretrained("./fake_news_model")
            tokenizer.save_pretrained("./fake_news_model")
        
        # Print classification report
        if epoch == EPOCHS:
            print("\n" + "="*50)
            print("FINAL EVALUATION METRICS")
            print("="*50)
            print("\nClassification Report:")
            print(classification_report(
                true_labels, 
                predictions,
                target_names=['Fake', 'Real']
            ))
            print("\nConfusion Matrix:")
            print(confusion_matrix(true_labels, predictions))
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: ./fake_news_model")
    print("="*50)

if __name__ == "__main__":
    main()