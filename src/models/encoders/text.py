"""Text encoder using transformer models for text encoding.

Optimizations for training speed:
1. Batch tokenization with padding
2. Optional text embedding caching
3. Reduced max_length for faster processing
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """文本编码器: 文本 → 256维 (使用预训练Transformer模型)
    
    Uses a pretrained transformer model as backbone for text encoding.
    Default is XLM-RoBERTa for cross-lingual support.
    Supports freezing backbone weights during training.
    
    Performance optimizations:
    - Embedding cache to avoid redundant computation
    - Efficient batch processing
    - Configurable max_length for speed/quality tradeoff
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dim: int = 256,
        max_length: int = 128,  # 减小默认值，加速处理
        freeze_backbone: bool = True,
        use_cache: bool = True  # 启用缓存
    ):
        """
        Args:
            model_name: Pretrained model name (default: xlm-roberta-base)
            output_dim: Output embedding dimension (default: 256)
            max_length: Maximum token length (default: 128, reduced for speed)
            freeze_backbone: Whether to freeze backbone weights (default: True)
            use_cache: Whether to cache embeddings (default: True)
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self._freeze_backbone = freeze_backbone
        self.use_cache = use_cache
        
        # Load pretrained model and tokenizer using Auto classes for flexibility
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from backbone config
        self.hidden_size = self.backbone.config.hidden_size
        
        # Projection layer: hidden_size → output_dim
        self.projection = nn.Linear(self.hidden_size, output_dim)
        
        # Embedding cache: text_hash -> embedding
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
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
        # Clear cache when unfreezing (embeddings will change)
        self.clear_cache()
    
    @property
    def is_backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._freeze_backbone
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def _get_text_hash(self, text: str) -> int:
        """Get hash for text string."""
        return hash(text)
    
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
        
        # Use cache if enabled and backbone is frozen
        if self.use_cache and self._freeze_backbone and not self.training:
            return self._forward_with_cache(processed_texts, empty_mask, device)
        
        # Standard forward pass
        return self._forward_batch(processed_texts, empty_mask, device)
    
    def _forward_with_cache(
        self, 
        texts: List[str], 
        empty_mask: List[bool],
        device: torch.device
    ) -> torch.Tensor:
        """Forward pass with caching for frozen backbone."""
        batch_size = len(texts)
        embeddings = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Find texts that need computation
        texts_to_compute = []
        indices_to_compute = []
        
        for i, (text, is_empty) in enumerate(zip(texts, empty_mask)):
            if is_empty:
                continue
            
            text_hash = self._get_text_hash(text)
            if text_hash in self._cache:
                embeddings[i] = self._cache[text_hash].to(device)
                self._cache_hits += 1
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
                self._cache_misses += 1
        
        # Compute embeddings for cache misses
        if texts_to_compute:
            computed = self._forward_batch(
                texts_to_compute, 
                [False] * len(texts_to_compute),
                device
            )
            
            # Store in cache and update output
            for j, (text, idx) in enumerate(zip(texts_to_compute, indices_to_compute)):
                text_hash = self._get_text_hash(text)
                self._cache[text_hash] = computed[j].detach().cpu()
                embeddings[idx] = computed[j]
        
        return embeddings
    
    def _forward_batch(
        self, 
        texts: List[str], 
        empty_mask: List[bool],
        device: torch.device
    ) -> torch.Tensor:
        """Standard batch forward pass."""
        batch_size = len(texts)
        
        # Tokenize texts with truncation
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Forward through backbone
        if self._freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
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
