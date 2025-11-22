"""
Stage 1: Internal State Interpretation
Interpretation methods for understanding model behavior through saliency and attention.

This module implements attribution methods to understand which tokens contribute
most to the model's predictions, using attention weights and integrated gradients.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr._utils.attribution import Attribution

import config


class ModelInterpreter:
    """
    Interpreter for BERT-based classification models.
    
    Provides methods to extract internal representations (attention) and
    compute token-level attributions using Integrated Gradients.
    
    Integrated Gradients (IG) computes the attribution of input features by
    integrating the gradients along the path from a baseline (typically zeros)
    to the input. The attribution for token i is:
    
        IG_i = (x_i - x'_i) * ∫[α=0 to 1] ∂F(x' + α(x - x'))/∂x_i dα
    
    where:
        - x is the input embedding
        - x' is the baseline embedding (zero)
        - F is the model function
        - α parameterizes the interpolation path
    """
    
    def __init__(self) -> None:
        """
        Initialize the ModelInterpreter by loading the BERT model and tokenizer.
        
        The model is loaded from config.MODEL_NAME, set to evaluation mode,
        and moved to the device specified in config.DEVICE.
        """
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        
        # Set model to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize Integrated Gradients attributor
        # Target layer: embeddings (input layer)
        # This will compute attributions for the embedding layer
        self.lig = LayerIntegratedGradients(
            forward_func=self._forward_func,
            layer=self.model.bert.embeddings
        )
        
        # Storage for current input context (used in forward function)
        self._current_attention_mask: Optional[torch.Tensor] = None
    
    def _forward_func(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward function for Integrated Gradients computation.
        
        This function is used by Captum's LayerIntegratedGradients. It receives
        the embeddings (output of the target layer) and continues the forward
        pass through the rest of the model.
        
        When LayerIntegratedGradients interpolates between baseline and input
        embeddings, this function is called with the interpolated embeddings.
        
        Args:
            embeddings: Interpolated embeddings tensor of shape 
                       (batch_size, seq_len, hidden_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Continue forward pass from embeddings through BERT
        outputs = self.model.bert(
            inputs_embeds=embeddings,
            attention_mask=self._current_attention_mask
        )
        
        # Get pooled output and pass through classifier
        pooled_output = outputs.pooler_output
        logits = self.model.classifier(pooled_output)
        
        return logits
    
    def predict(self, text: str) -> Tuple[torch.Tensor, int]:
        """
        Predict the class and return probability distribution.
        
        The probability distribution is computed using softmax:
            P(y_i | x) = exp(logit_i) / Σ_j exp(logit_j)
        
        Args:
            text: Input text string to classify
            
        Returns:
            A tuple containing:
                - probs: Probability distribution tensor of shape (num_classes,)
                - predicted_class: Index of the predicted class (int)
        """
        # Tokenize input
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get predicted class
        predicted_class = torch.argmax(probs, dim=-1).item()
        
        return probs[0], predicted_class
    
    def get_attentions(self, text: str) -> np.ndarray:
        """
        Extract attention weights from the last transformer layer.
        
        Attention weights represent how much each token attends to every other token.
        For multi-head attention, we average across all heads:
            Attention_avg = (1/H) * Σ_h Attention_h
        
        The attention weights are normalized per token (row-wise):
            Attention_ij = exp(score_ij) / Σ_k exp(score_ik)
        
        Args:
            text: Input text string
            
        Returns:
            Average attention weights as numpy array of shape (seq_len, seq_len)
            where attention[i, j] represents how much token i attends to token j.
        """
        # Tokenize input
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get attention from all layers (outputs.attentions is a tuple of tensors)
        # Each tensor has shape (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        
        # Get the last layer's attention
        last_layer_attention = attentions[-1]  # Shape: (1, num_heads, seq_len, seq_len)
        
        # Average across heads: (1, num_heads, seq_len, seq_len) -> (1, seq_len, seq_len)
        averaged_attention = torch.mean(last_layer_attention, dim=1)
        
        # Remove batch dimension: (1, seq_len, seq_len) -> (seq_len, seq_len)
        averaged_attention = averaged_attention[0]
        
        # Convert to numpy
        attention_np = averaged_attention.cpu().numpy()
        
        return attention_np
    
    def compute_integrated_gradients(
        self, 
        text: str, 
        target_class: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Compute Integrated Gradients attributions for each token.
        
        Integrated Gradients attributes the model's prediction to input features
        by integrating gradients along the path from a baseline (zero embeddings)
        to the actual input. The attribution for token i is:
        
            IG_i = (x_i - baseline_i) * ∫[α=0 to 1] ∂F/∂x_i(x' + α(x - x')) dα
        
        where the integral is approximated using the trapezoidal rule with n_steps.
        
        Args:
            text: Input text string
            target_class: Target class index for attribution. If None, uses the
                         predicted class from the model.
        
        Returns:
            List of dictionaries with keys 'token' and 'score', where:
                - 'token': Token string (excluding special tokens CLS, SEP)
                - 'score': Normalized attribution score (L2 norm = 1)
            
            The list is sorted by token position in the original text.
        """
        # Tokenize input
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Store attention mask for use in forward function
        self._current_attention_mask = attention_mask
        
        # Get target class if not provided
        if target_class is None:
            _, target_class = self.predict(text)
        
        # Baseline: use padding token (typically 0) for all positions
        baseline_ids = torch.zeros_like(input_ids)
        
        # Compute attributions
        # attributions shape: (batch_size, seq_len, hidden_size)
        # LayerIntegratedGradients computes attributions for the embedding layer
        # It will interpolate between baseline and input embeddings internally
        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=target_class,
            n_steps=50  # Number of steps for integral approximation
        )
        
        # Sum attributions across embedding dimension to get token-level scores
        # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len)
        token_attributions = attributions.sum(dim=-1)
        
        # Remove batch dimension: (batch_size, seq_len) -> (seq_len,)
        token_attributions = token_attributions[0]
        
        # Convert to numpy
        token_scores = token_attributions.cpu().detach().numpy()
        
        # Get tokens for mapping
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out special tokens and create result list
        result = []
        special_tokens = [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                         self.tokenizer.pad_token, '[CLS]', '[SEP]', '[PAD]']
        
        for token, score in zip(tokens, token_scores):
            if token not in special_tokens:
                result.append({'token': token, 'score': float(score)})
        
        # Normalize scores to unit L2 norm
        if result:
            scores_array = np.array([item['score'] for item in result])
            norm = np.linalg.norm(scores_array)
            
            if norm > 0:
                scores_array = scores_array / norm
                
                # Update scores in result
                for i, item in enumerate(result):
                    item['score'] = float(scores_array[i])
        
        return result
