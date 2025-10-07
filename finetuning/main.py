import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# LoRA Implementation
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, in_features) or (batch_size, in_features)
        lora_output = x @ self.lora_A
        lora_output = self.dropout(lora_output)
        lora_output = lora_output @ self.lora_B
        return lora_output * self.scaling

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear_layer: nn.Linear, rank: int = 16, alpha: float = 32, dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, 
            alpha, 
            dropout
        )
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora_to_model(model, rank: int = 16, alpha: float = 32, dropout: float = 0.1):
    """Apply LoRA to attention layers in DistilBERT"""
    lora_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in ['q_lin', 'k_lin', 'v_lin', 'out_lin']):
            # Get parent module
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            # Replace with LoRA version
            original_layer = getattr(parent, attrs[-1])
            lora_layer = LoRALinear(original_layer, rank, alpha, dropout)
            setattr(parent, attrs[-1], lora_layer)
            lora_layers.append(name)
    
    print(f"Applied LoRA to {len(lora_layers)} layers: {lora_layers}")
    return model

# Data Processing
class BiasInBiosDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
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
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor:
    def __init__(self, tokenizer, max_length: int = 512, test_size: float = 0.2, val_size: float = 0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
    
    def load_and_preprocess_data(self):
        """Load and preprocess the Bias in Bios dataset"""
        print("Loading Bias in Bios dataset...")
        
        # Load dataset
        train_dataset = load_dataset("LabHC/bias_in_bios", split='train')
        
        # Extract texts and labels
        texts = [item['hard_text'] for item in train_dataset]
        labels = [item['profession'] for item in train_dataset]
        
        # Get unique professions
        unique_labels = sorted(list(set(labels)))
        num_labels = len(unique_labels)
        
        # Create label mapping
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        # Convert labels to sequential integers
        labels = [label_to_id[label] for label in labels]
        
        print(f"Dataset loaded: {len(texts)} samples, {num_labels} professions")
        print(f"Sample text length: {np.mean([len(text.split()) for text in texts[:1000]]):.1f} words")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=self.test_size, random_state=42, stratify=labels
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = BiasInBiosDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = BiasInBiosDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = BiasInBiosDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, num_labels, id_to_label

# Training Pipeline
class DistilBERTLoRATrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        val_dataloader, 
        device,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        
        # Optimizer - only optimize LoRA parameters
        lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
        print(f"Number of LoRA parameters: {sum(p.numel() for p in lora_params):,}")
        
        self.optimizer = AdamW(lora_params, lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}
    
    def train(self) -> Dict[str, List[float]]:
        history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                  'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        best_val_f1 = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model('best_model.pt')
        
        return history
    
    def save_model(self, filepath: str):
        """Save LoRA weights and model configuration"""
        lora_state_dict = {k: v for k, v in self.model.state_dict().items() if 'lora' in k}
        torch.save({
            'lora_state_dict': lora_state_dict,
            'model_config': self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else None
        }, filepath)
        print(f"Model saved to {filepath}")

def load_model_with_lora(model_path: str, model_name: str = "distilbert-base-uncased", num_labels: int = 28):
    """Load model with LoRA weights"""
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = apply_lora_to_model(model)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['lora_state_dict'], strict=False)
    
    return model

def evaluate_model(model, test_dataloader, device, id_to_label):
    """Comprehensive evaluation on test set"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy, f1, predictions, true_labels

# Main execution pipeline
def main():
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Process data
    data_processor = DataProcessor(tokenizer, max_length=MAX_LENGTH)
    train_dataset, val_dataset, test_dataset, num_labels, id_to_label = data_processor.load_and_preprocess_data()
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    
    # Apply LoRA
    model = apply_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Train model
    trainer = DistilBERTLoRATrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )
    
    history = trainer.train()
    
    # Load best model and evaluate
    best_model = load_model_with_lora('best_model.pt', MODEL_NAME, num_labels)
    best_model.to(device)
    
    test_acc, test_f1, predictions, true_labels = evaluate_model(
        best_model, test_dataloader, device, id_to_label
    )
    
    # Save metadata
    metadata = {
        'num_labels': num_labels,
        'id_to_label': id_to_label,
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'lora_config': {
            'rank': LORA_RANK,
            'alpha': LORA_ALPHA,
            'dropout': LORA_DROPOUT
        },
        'test_accuracy': test_acc,
        'test_f1': test_f1
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining completed successfully!")
    print(f"Best model saved as 'best_model.pt'")
    print(f"Metadata saved as 'model_metadata.json'")
    
    return model, history, metadata

# Example inference function
def predict_profession(text: str, model_path: str = 'best_model.pt', metadata_path: str = 'model_metadata.json'):
    """Predict profession for a given biography text"""
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = load_model_with_lora(
        model_path, 
        metadata['model_name'], 
        metadata['num_labels']
    )
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(metadata['model_name'])
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=metadata['max_length'],
        return_tensors='pt'
    )
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    predicted_profession = metadata['id_to_label'][str(predicted_class)]
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_profession, confidence

if __name__ == "__main__":
    model, history, metadata = main()
    
    # Example inference
    sample_text = "She is able to assess, diagnose and treat minor illness conditions and exacerbations of some long term conditions. Her qualifications include Registered General Nurse, Bachelor of Nursing, Diploma in Health Science, Emergency Care Practitioner and Independent Nurse Prescribing."
    
    profession, confidence = predict_profession(sample_text)
    print(f"\nExample prediction:")
    print(f"Text: {sample_text[:100]}...")
    print(f"Predicted profession: {profession}")
    print(f"Confidence: {confidence:.4f}")