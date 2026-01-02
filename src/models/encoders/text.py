"""Text encoder using transformer models for text encoding."""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """文本编码器: 文本 → 256维 (使用预训练Transformer模型)
    
    Uses a pretrained transformer model as backbone for text encoding.
    Default is XLM-RoBERTa for cross-lingual support.
    Supports freezing backbone weights during training.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dim: int = 256,
        max_length: int = 512,
        freeze_backbone: bool = True
    ):
        """
        Args:
            model_name: Pretrained model name (default: xlm-roberta-base)
            output_dim: Output embedding dimension (default: 256)
            max_length: Maximum token length (default: 512)
            freeze_backbone: Whether to freeze backbone weights (default: True)
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self._freeze_backbone = freeze_backbone
        
        # Load pretrained model and tokenizer using Auto classes for flexibility
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from backbone config
        self.hidden_size = self.backbone.config.hidden_size
        
        # Projection layer: hidden_size → output_dim
        self.projection = nn.Linear(self.hidden_size, output_dim)
        
        # Apply initial freeze setting
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self) -> None:
        """Freeze backbone weights (only projection layer will be trained)."""
        self._freeze_backbone = True
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights (all parameters will be trained)."""
        self._freeze_backbone = False
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    @property
    def is_backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._freeze_backbone
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text strings into embeddings.
        
        Args:
            texts: List of text strings [batch]
            
        Returns:
            Tensor[batch, output_dim] text embeddings
        """
        batch_size = len(texts)
        device = self.projection.weight.device
        
        # Handle empty batch
        if batch_size == 0:
            return torch.zeros(0, self.output_dim, device=device)
        
        # Check for empty/None texts and replace with empty string
        processed_texts = []
        empty_mask = []
        for text in texts:
            if text is None or text == "" or (isinstance(text, str) and text.strip() == ""):
                processed_texts.append("")
                empty_mask.append(True)
            else:
                processed_texts.append(text)
                empty_mask.append(False)
        
        # If all texts are empty, return zero embeddings
        if all(empty_mask):
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        # Tokenize texts with truncation
        encoding = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Forward through backbone
        with torch.set_grad_enabled(not self._freeze_backbone or self.training):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to output dimension
        embeddings = self.projection(cls_output)
        
        # Zero out embeddings for empty texts
        empty_mask_tensor = torch.tensor(empty_mask, device=device).unsqueeze(1)
        embeddings = embeddings * (~empty_mask_tensor).float()
        
        return embeddings
