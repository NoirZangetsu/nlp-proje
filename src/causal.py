"""
Stage 2: Causal Tracing
Causal analysis through counterfactual generation.

This module implements causal tracing by generating counterfactuals to distinguish
features with true causal impact from those that are merely correlated.

The Causal Impact Score (CIS) measures the change in prediction probability when
a token is masked, isolating features that truly cause the model's prediction.
"""
from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F

from src.interpretation import ModelInterpreter
import config


class CausalTracer:
    """
    Causal tracer that validates saliency scores through counterfactual interventions.
    
    This class implements the methodology to filter features based on causality rather
    than just correlation. By masking important tokens and observing the change in
    prediction probability, we can identify which tokens have true causal impact.
    
    The Causal Impact Score (CIS) is defined as:
        CIS = P(y_pred | x_original) - P(y_pred | x_masked)
    
    where:
        - x_original is the original input
        - x_masked is the input with the token masked
        - y_pred is the originally predicted class
    
    A high positive CIS indicates that the token was causally necessary for the
    prediction, while a low or negative CIS suggests the token was not critical.
    """
    
    def __init__(self, interpreter: ModelInterpreter) -> None:
        """
        Initialize the CausalTracer with a ModelInterpreter instance.
        
        Args:
            interpreter: An instance of ModelInterpreter that provides prediction
                        and attribution methods.
        """
        self.interpreter = interpreter
        self.tokenizer = interpreter.tokenizer
        self.device = interpreter.device
        
        # Get mask token ID (use [UNK] if available, otherwise [PAD])
        if self.tokenizer.unk_token_id is not None:
            self.mask_token_id = self.tokenizer.unk_token_id
        else:
            self.mask_token_id = self.tokenizer.pad_token_id
    
    def _create_counterfactual_input_ids(
        self, 
        text: str, 
        token_to_mask: str, 
        token_position: int
    ) -> torch.Tensor:
        """
        Create counterfactual input IDs by masking a specific token position.
        
        This method tokenizes the text and replaces the token at the specified
        position with a mask token ID, returning the modified input_ids.
        
        Args:
            text: Original input text
            token_to_mask: The token string to mask (for verification)
            token_position: Position of the token in the tokenized sequence
            
        Returns:
            Modified input_ids tensor with the token masked
        """
        # Tokenize the original text
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        input_ids = encoded["input_ids"][0].clone()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Verify the token at the position matches
        if token_position < len(tokens) and tokens[token_position] == token_to_mask:
            # Replace with mask token
            input_ids[token_position] = self.mask_token_id
        
        return input_ids.unsqueeze(0)  # Add batch dimension
    
    def _find_token_position(
        self, 
        text: str, 
        target_token: str
    ) -> Optional[int]:
        """
        Find the position of a token in the tokenized sequence.
        
        Args:
            text: Original input text
            target_token: Token string to find
            
        Returns:
            Position index if found, None otherwise
        """
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        
        # Find first occurrence of the token
        for i, token in enumerate(tokens):
            if token == target_token:
                return i
        
        return None
    
    def generate_counterfactuals(
        self, 
        text: str, 
        top_n_tokens: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactuals for top important tokens and compute causal impact.
        
        This method implements the causal tracing pipeline:
        1. Get saliency scores via Integrated Gradients
        2. Select top N tokens with highest positive scores
        3. For each token, create a counterfactual by masking it
        4. Compute Causal Impact Score (CIS)
        5. Validate tokens with CIS > threshold
        
        Args:
            text: Input text string to analyze
            top_n_tokens: Number of top tokens to test (default: 5)
        
        Returns:
            List of dictionaries with keys:
                - 'token': Token string
                - 'saliency': Original saliency score from Integrated Gradients
                - 'causal_impact': Causal Impact Score (CIS)
                - 'is_validated': Boolean indicating if CIS > 0.1
        """
        # Step 1: Get original prediction and saliency scores
        original_probs, predicted_class = self.interpreter.predict(text)
        original_prob = original_probs[predicted_class].item()
        
        # Get integrated gradients attributions
        attributions = self.interpreter.compute_integrated_gradients(text)
        
        # Step 2: Select top N tokens with highest positive scores
        # Filter for positive scores only (tokens that increase prediction confidence)
        positive_attributions = [
            attr for attr in attributions 
            if attr['score'] > 0
        ]
        
        # Sort by score (descending) and take top N
        sorted_attributions = sorted(
            positive_attributions, 
            key=lambda x: x['score'], 
            reverse=True
        )[:top_n_tokens]
        
        if not sorted_attributions:
            # No positive attributions found
            return []
        
        # Step 3: Intervention loop - test each top token
        results = []
        cis_threshold = 0.1  # Threshold for validation
        
        for attr in sorted_attributions:
            token = attr['token']
            saliency = attr['score']
            
            # Find token position in original text
            token_position = self._find_token_position(text, token)
            
            if token_position is None:
                # Token not found (shouldn't happen, but handle gracefully)
                results.append({
                    'token': token,
                    'saliency': saliency,
                    'causal_impact': 0.0,
                    'is_validated': False
                })
                continue
            
            # Create counterfactual input IDs by masking the token
            counterfactual_input_ids = self._create_counterfactual_input_ids(
                text, 
                token, 
                token_position
            )
            
            # Get prediction for counterfactual
            try:
                # Tokenize to get attention mask
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_LENGTH
                )
                attention_mask = encoded["attention_mask"].to(self.device)
                counterfactual_input_ids = counterfactual_input_ids.to(self.device)
                
                # Forward pass with counterfactual
                with torch.no_grad():
                    outputs = self.interpreter.model(
                        input_ids=counterfactual_input_ids,
                        attention_mask=attention_mask
                    )
                    counterfactual_logits = outputs.logits
                
                # Compute softmax probabilities
                counterfactual_probs = F.softmax(counterfactual_logits, dim=-1)
                counterfactual_prob = counterfactual_probs[0][predicted_class].item()
                
                # Step 4: Calculate Causal Impact Score
                # CIS = Original_Probability - Counterfactual_Probability
                causal_impact = original_prob - counterfactual_prob
                
                # Step 5: Validate (CIS > threshold)
                is_validated = causal_impact > cis_threshold
                
                results.append({
                    'token': token,
                    'saliency': saliency,
                    'causal_impact': causal_impact,
                    'is_validated': is_validated
                })
                
            except Exception as e:
                # Handle any errors during counterfactual prediction
                # (e.g., if masking creates invalid input)
                results.append({
                    'token': token,
                    'saliency': saliency,
                    'causal_impact': 0.0,
                    'is_validated': False
                })
        
        return results
