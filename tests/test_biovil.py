import os
import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BatchEncoding

# Import classes and functions from your script file
from train import (
    BiovilDataset,
    VisualProjectionLayer,
    CrossAttentionClassifierBiovil,
    get_text_embeddings,
    get_image_embeddings,
    generate_encoded_tensors,
    load_saved_images,
    load_saved_texts,
    cross_attention_train_biovil,
)


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def sample_df():
    """Provides a dummy pandas DataFrame matching CheXpert structure."""
    return pd.DataFrame({
        "path_to_image": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "report": ["No acute disease.", "Mild cardiomegaly.", "Pleural effusion."]
    })


@pytest.fixture
def mock_mlflow():
    """Mocks mlflow logging calls to avoid needing an active MLflow server."""
    with patch("train.mlflow") as mock:
        yield mock


# ==========================================
# DATASET TESTS
# ==========================================

class TestBiovilDataset:
    def test_len(self, sample_df):
        dataset = BiovilDataset(sample_df)
        assert len(dataset) == len(sample_df)

    def test_getitem(self, sample_df):
        base_path = "./custom_data/"
        dataset = BiovilDataset(sample_df, base_path=base_path)
        img_path, report, idx = dataset[0]

        assert img_path == "./custom_data/img1.jpg"
        assert report == "No acute disease."
        assert idx == 0


# ==========================================
# MODEL LAYER TESTS
# ==========================================

class TestVisualProjectionLayer:
    def test_output_shape(self):
        batch_size = 2
        img_dim = 128
        h, w = 14, 14
        text_dim = 768

        layer = VisualProjectionLayer(img_dim=img_dim, text_dim=text_dim)
        # Expected input shape: [batch_size, 128, 14, 14]
        dummy_input = torch.randn(batch_size, img_dim, h, w)
        output = layer(dummy_input)

        # Output should be permuted & flattened spatial tokens: [batch_size, 14*14, 768]
        assert output.shape == (batch_size, h * w, text_dim)


class TestCrossAttentionClassifierBiovil:
    def test_forward_pass(self):
        model = CrossAttentionClassifierBiovil(embed_dim=768, num_heads=8)
        model.eval()

        # [batch_size, img_dim, H, W]
        dummy_img = torch.randn(2, 128, 14, 14)
        # [batch_size, seq_len, embed_dim]
        dummy_txt = torch.randn(2, 256, 768)

        with torch.no_grad():
            output = model(dummy_img, dummy_txt)

        # Output shape should be [batch_size, 1] for binary classification
        assert output.shape == (2, 1)


# ==========================================
# FEATURE EXTRACTION & EMBEDDING TESTS
# ==========================================

class TestEmbeddingFunctions:
    @patch("train.Image.open")
    def test_get_image_embeddings(self, mock_image_open):
        # Mock PIL image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        # Mock image transform and model
        mock_transform = MagicMock(return_value=torch.randn(1, 448, 448))
        mock_model = MagicMock()
        
        # Setup model return structure
        mock_output = MagicMock()
        mock_output.projected_patch_embeddings = torch.randn(1, 128, 14, 14)
        mock_model.return_value = mock_output

        res = get_image_embeddings("fake_path.jpg", mock_transform, mock_model)

        assert res.shape == (1, 128, 14, 14)
        mock_image_open.assert_called_once_with("fake_path.jpg")

    def test_get_text_embeddings(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = BatchEncoding({
            "input_ids": torch.randint(0, 1000, (1, 512)),
            "attention_mask": torch.ones((1, 512))
        })

        mock_text_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 512, 768)
        mock_text_model.return_value = mock_output

        res = get_text_embeddings("Sample report text", mock_tokenizer, mock_text_model)

        assert res.shape == (1, 512, 768)


# ==========================================
# CHUNKING & I/O TESTS
# ==========================================

class TestChunkingAndStorage:
    @patch("train.get_text_embeddings")
    @patch("train.get_image_embeddings")
    @patch("train.create_chest_xray_transform_for_inference")
    @patch("train.get_biovil_t_image_encoder")
    @patch("train.AutoModel.from_pretrained")
    @patch("train.AutoTokenizer.from_pretrained")
    def test_generate_encoded_tensors_and_loaders(
        self,
        mock_tokenizer,
        mock_auto_model,
        mock_get_encoder,
        mock_transform,
        mock_get_img_emb,
        mock_get_txt_emb,
        sample_df,
        tmp_path,
        monkeypatch
    ):
        # Redirect working directory to isolated temp path to catch disk dumps
        monkeypatch.chdir(tmp_path)

        # Mock embedding return shapes
        mock_get_img_emb.return_value = torch.randn(1, 128, 14, 14)
        mock_get_txt_emb.return_value = torch.randn(1, 512, 768)

        # Generate chunk files with small chunk_size=2
        last_chunk_idx = generate_encoded_tensors(sample_df, type="unittest", chunk_size=2)

        # 3 items with chunk size 2 -> 2 chunks total (idx 0 and 1)
        assert last_chunk_idx == 1

        # Test loading saved tensors back into memory
        loaded_imgs = load_saved_images(last_chunk_idx, type="unittest")
        loaded_txts = load_saved_texts(last_chunk_idx, type="unittest")

        assert len(loaded_imgs) == len(sample_df)
        assert len(loaded_txts) == len(sample_df)
        # Check text sequence slicing [:256, :] logic from load_saved_texts
        assert loaded_txts[0].shape == (256, 768)
        assert loaded_imgs[0].shape == (128, 14, 14)


# ==========================================
# TRAINING LOOP TESTS
# ==========================================

class TestTrainingLoop:
    def test_cross_attention_train_biovil_runs_and_stops(self, mock_mlflow):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create minimal synthetic Dataset
        img_data = torch.randn(10, 128, 14, 14)
        txt_data = torch.randn(10, 256, 768)
        labels = torch.randint(0, 2, (10,)).float()

        dataset = TensorDataset(img_data, txt_data, labels)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        # Setup light components
        model = CrossAttentionClassifierBiovil().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

        history = cross_attention_train_biovil(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=3,
            patience=2
        )

        # Verify history output structure
        assert "loss" in history
        assert "val_loss" in history
        assert len(history["loss"]) > 0
        assert mock_mlflow.log_metric.called