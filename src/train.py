import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import copy
import argparse

from sklearn.model_selection import train_test_split
import torch
from PIL import Image   
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

import mlflow
import mlflow.pytorch

# CLASSES
class BiovilDataset(Dataset):
    def __init__(self, df, base_path="./data/CheXpert_Images/"):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.base_path + self.df.loc[idx, "path_to_image"]
        report = self.df.loc[idx, "report"]
        return img_path, report, idx

class VisualProjectionLayer(nn.Module):
    def __init__(self, img_dim=128, text_dim=768):
        super().__init__()
        # This layer bridges the dimensionality gap (128 -> 768)
        self.projector = nn.Linear(img_dim, text_dim)
        
    def forward(self, img_patches):
        # img_patches is [128, 14, 14]
        # Move channels to the back: [14, 14, 128]
        x = img_patches.permute(0,2, 3, 1)
        
        # Flatten spatial dimensions: [196, 128]
        x = x.flatten(1, 2)
        
        # 3. Project to text dimension: [196, 768]
        projected_visual_tokens = self.projector(x)
        return projected_visual_tokens

class CrossAttentionClassifierBiovil(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.projection_layer = VisualProjectionLayer()

        self.cross_attention = nn.MultiheadAttention(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    dropout=dropout,
                    batch_first=True
                )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
        # Add a small Feed-Forward Network layer (FFN) inside the attention block 
        # This mirrors a true standard Transformer block and stabilizes representation warping
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, img_patches, text_tokens):
        img_patches_proj = self.projection_layer(img_patches)
        norm_text = self.layer_norm1(text_tokens)
        # Cross-Attention Core
        attn_output, _ = self.cross_attention(
            query=norm_text, 
            key=img_patches_proj, 
            value=img_patches_proj
        )
        x = (attn_output + text_tokens)
        
        # FFN stabilization step
        x = self.layer_norm2(self.ffn(x)) + x
        
        # Step 3: Max-pooling across sequence dimension 
        # This captures the strongest text-visual features and ignores inactive padding tokens
        x_pooled, _ = torch.max(x, dim=1)
        # Step 4: Predict
        return self.classifier(x_pooled)

# FUNCTIONS
def get_text_embeddings(report_text, tokenizer, text_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tokenize standard CheXpert format
    inputs = tokenizer(
        report_text, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        #  To get the raw token hidden states for Cross-Attention
        outputs = text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True
        )
        # Pull out the hidden states sequence
        sequence_outputs = outputs.last_hidden_state
        
    return sequence_outputs # Shape: [1, 512, 768]

def get_image_embeddings(image_path, image_transform, image_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Extracts true BioViL-T spatial image embeddings.
    raw_image = Image.open(image_path).convert("L")
    processed_tensor = image_transform(raw_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get the native image model outputs
        image_outputs = image_model(processed_tensor)
        
        # Pull global 128-d vector and spatial patch tokens
        patch_img_emb = image_outputs.projected_patch_embeddings 
        
    return patch_img_emb # Shape: [1, 128, 14, 14]

def generate_encoded_tensors(x_set, type="train", chunk_size=2000):
    # Setup device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the comprehensive BioViL-T repo for Text
    model_id = "microsoft/BiomedVLP-BioViL-T"
    # Specialized CXR-BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    # Instantiate the BioViL-T Image Engine
    image_model = get_biovil_t_image_encoder().to(device)
    image_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)

    # Setup folders to store heavy tensors for Phase 2 Cross-Attention
    os.makedirs(f"./data/features_chunks/image_patches_{type}", exist_ok=True)
    os.makedirs(f"./data/features_chunks/text_sequences_{type}", exist_ok=True)

    train_dataset = BiovilDataset(x_set)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    img_buffer = {}
    text_buffer = {}
    chunk_idx = 0

    print("Extracting features")
    for img_paths, reports, indices in tqdm(train_loader):
        img_path = img_paths[0]
        report = reports[0]
        idx = indices[0].item()
        
        # Extract encoded vectors
        img_patches = get_image_embeddings(img_path, image_transform, image_model)      
        text_sequence = get_text_embeddings(report, tokenizer, text_model)

        # Cast to float16 and squeeze batch dimension [1, ...] -> [...]
        # Store directly in memory buffers keyed by sample index
        img_buffer[idx] = img_patches.detach().to(torch.float16).cpu()
        text_buffer[idx] = text_sequence.detach().to(torch.float16).cpu()    

        # Save chunk to disk every CHUNK_SIZE items
        if len(img_buffer) >= chunk_size:
            torch.save(img_buffer, f"./data/features_chunks/image_patches_{type}/img_patches_chunk_{chunk_idx}.pt")
            torch.save(text_buffer, f"./data/features_chunks/text_sequences_{type}/text_seq_chunk_{chunk_idx}.pt")
            # Reset buffers for next chunk
            img_buffer.clear()
            text_buffer.clear()
            chunk_idx += 1   
    
    # Save remaining items after loop completes
    if len(img_buffer) > 0:
        torch.save(img_buffer, f"./data/features_chunks/image_patches_{type}/img_patches_chunk_{chunk_idx}.pt")
        torch.save(text_buffer, f"./data/features_chunks/text_sequences_{type}/text_seq_chunk_{chunk_idx}.pt")
        img_buffer.clear()
        text_buffer.clear()

    return chunk_idx

def load_saved_images(chunk_idx, type='train'):
    ret_array = []
    # Load image patches: expected shape [128, 14, 14]
    for i in tqdm(range(chunk_idx+1)):
        img_patches = torch.load(f"./data/features_chunks/image_patches_{type}/img_patches_chunk_{i}.pt", weights_only=True)
        ret_array.extend([img_patches[j].squeeze(0) for j in sorted(img_patches.keys())])

    return ret_array
    
def load_saved_texts(chunk_idx, type='train'):
    ret_array = []    
    # Load text sequence: expected shape [512, 768]
    for i in tqdm(range(chunk_idx+1)):
        text_sequence = torch.load(f"./data/features_chunks/text_sequences_{type}/text_seq_chunk_{i}.pt", weights_only=True)
        ret_array.extend([text_sequence[j].squeeze(0)[:256, :] for j in sorted(text_sequence.keys())])

    return ret_array

def cross_attention_train_biovil(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=3):
    """
    Function to train a PyTorch model with training and validation datasets.

    Parameters:
    model: The neural network model to train.
    train_loader: DataLoader for the training dataset.
    val_loader: DataLoader for the validation dataset.
    criterion: Loss function (e.g., Binary Cross Entropy for classification).
    optimizer: Optimization algorithm (e.g., Adam, SGD).
    epochs: Number of training epochs (default=100).

    Returns:
    history: Dictionary containing loss and accuracy for both training and validation.
    """
    # Setup device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dictionary to store training & validation loss and accuracy over epochs
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    # Early stopping trackers
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    best_epoch = 0

    for epoch in range(epochs):  # Loop over the number of epochs
        model.train()  # Set model to training mode
        total_loss, correct = 0, 0  # Initialize total loss and correct predictions

        # Training loop
        for inputs_img, inputs_txt, labels in train_loader:
            inputs_img = inputs_img.to(device)
            inputs_txt = inputs_txt.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # Reset gradients before each batch
            outputs = model(inputs_img, inputs_txt).squeeze(1)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation (compute gradients)
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate batch loss
            correct += ((torch.sigmoid(outputs) >= 0.5).float() == labels).sum().item()  # Count correct predictions

        # Compute average loss and accuracy for training
        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # Validation phase (without gradient computation)
        model.eval()  # Set model to evaluation mode
        val_loss, val_correct = 0, 0
        with torch.no_grad():  # No need to compute gradients during validation
            for inputs_img, inputs_txt, labels in val_loader:
                inputs_img = inputs_img.to(device)
                inputs_txt = inputs_txt.to(device)
                labels = labels.to(device)
                outputs = model(inputs_img, inputs_txt).squeeze(1) # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                val_loss += loss.item()  # Accumulate validation loss
                val_correct += ((torch.sigmoid(outputs) >= 0.5).float() == labels).sum().item()  # Count correct predictions

        # Compute average loss and accuracy for validation
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        # Track metrics so they show up in Hugging Face / MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        #scheduler.step()
        scheduler.step(val_loss)
        # Capture current learning rate for tracking
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Store metrics in history dictionary
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        # Print training progress
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter since we found a better model
            best_epoch = epoch + 1
            # Cache a deep copy of the optimal weights
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"-> Validation loss did not improve. Early Stopping Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered! Stopping training at epoch {epoch+1}.")
            break

    # --- Restore Best Weights ---
    if best_model_weights is not None:
        print(f"✅ Restoring best model weights found at Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_weights)

    return history  # Return training history

if __name__ == "__main__":

    # Parse Model Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--experiment_name", type=str, default="lead_demo_training")
    args = parser.parse_args()

    # Load the dataset
    dataset_sample = pd.read_csv("./data/chexpert_plus_dataset_sample.csv", index_col=0)
    dataset_sample = dataset_sample.loc[0:20000, :]
    # Split the dataset into train and test sets
    X = dataset_sample.drop(columns=["target"])
    y = dataset_sample["target"]
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Get images/reports encoded tensors, outputs of BioVil-t
    chunks_total_train = generate_encoded_tensors(X_train, type="train")
    chunks_total_test = generate_encoded_tensors(X_test, type="test")

    # Construct datasets and dataloaders for train and test
    print("Loading encoded tesnors")
    cross_att_X_train_img_tensor = torch.stack(load_saved_images(chunks_total_train, type="train")).float()
    cross_att_X_train_txt_tensor = torch.stack(load_saved_texts(chunks_total_train, type="train")).float()
    cross_att_y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    cross_att_train_dataset = TensorDataset(cross_att_X_train_img_tensor, cross_att_X_train_txt_tensor, cross_att_y_train_tensor)

    cross_att_X_test_img_tensor = torch.stack(load_saved_images(chunks_total_test, type="test")).float()
    cross_att_X_test_txt_tensor = torch.stack(load_saved_texts(chunks_total_test, type="test")).float()
    cross_att_y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    cross_att_test_dataset = TensorDataset(cross_att_X_test_img_tensor, cross_att_X_test_txt_tensor, cross_att_y_test_tensor)

    cross_att_train_loader = DataLoader(cross_att_train_dataset, batch_size=64, shuffle=True)
    cross_att_val_loader = DataLoader(cross_att_test_dataset, batch_size=64, shuffle=False)

    print("Training start")
    # Train the model
    # Set tracking URI to Hugging Face server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cross_att_model = CrossAttentionClassifierBiovil().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(cross_att_model.parameters(), lr=1e-5, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        # Clear cache just to be safe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log hyperparams explicitly
        mlflow.log_params({
            "lr": 1e-5,
            "weight_decay": 1e-2,
            "epochs": args.n_epochs,
            "scheduler": "ReduceLROnPlateau"
        })

        # Launch the training block
        print("Training start")
        history = cross_attention_train_biovil(
            model=cross_att_model, 
            train_loader=cross_att_train_loader, 
            val_loader=cross_att_val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler = scheduler,
            epochs=args.n_epochs
        )

        # Log the best model into MLflow models
        mlflow.pytorch.log_model(cross_att_model, name="biovil_t_lead_demo")
