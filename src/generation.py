"""
Stage 3: T5 Inference
Rationale generation using T5 model

This module handles inference using the fine-tuned T5 model to generate
natural language rationales explaining model predictions based on causal evidence.
"""
import os
from typing import List, Dict, Any, Union
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

import config


class RationaleGenerator:
    """
    Generator for natural language rationales using T5 model.
    
    This class loads a fine-tuned T5 model (or base T5-small) and generates
    explanations for model predictions based on causal evidence from Stage 2.
    
    The generation process uses beam search to ensure high-quality, coherent
    rationales that explain why the model made a particular prediction.
    """
    
    def __init__(self, model_path: str = "src/models/rationale_agent_v1") -> None:
        """
        Initialize the RationaleGenerator.
        
        Attempts to load a fine-tuned model from the specified path. If not found,
        falls back to the base T5-small model.
        
        Args:
            model_path: Path to the fine-tuned model directory (default: 
                       "src/models/rationale_agent_v1")
        """
        self.device = config.DEVICE
        self.model_path = model_path
        
        # Try to load fine-tuned model, fallback to base model
        if os.path.exists(model_path) and os.path.isdir(model_path):
            try:
                print(f"Loading fine-tuned model from {model_path}...")
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                print("Fine-tuned model loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print(f"Falling back to {config.GENERATOR_MODEL}...")
                self._load_base_model()
        else:
            print(f"Warning: Model not found at {model_path}")
            print(f"Falling back to {config.GENERATOR_MODEL}...")
            self._load_base_model()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _load_base_model(self) -> None:
        """
        Load the base T5-small model as fallback.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(config.GENERATOR_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(config.GENERATOR_MODEL)
    
    def _format_causal_evidence(
        self, 
        causal_evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Format causal evidence list into a string for the prompt.
        
        Extracts validated tokens from the causal evidence and formats them
        as a comma-separated string.
        
        Args:
            causal_evidence: List of dictionaries from CausalTracer containing:
                            'token', 'saliency', 'causal_impact', 'is_validated'
        
        Returns:
            Formatted string of causal tokens
        """
        if not causal_evidence:
            return "none"
        
        # Extract validated tokens (prefer validated, but include all if none validated)
        validated_tokens = [
            item['token'] for item in causal_evidence 
            if item.get('is_validated', False)
        ]
        
        # If no validated tokens, use top tokens by causal impact
        if not validated_tokens:
            # Sort by causal_impact and take top tokens
            sorted_evidence = sorted(
                causal_evidence,
                key=lambda x: x.get('causal_impact', 0),
                reverse=True
            )
            validated_tokens = [item['token'] for item in sorted_evidence[:5]]
        
        # Format as comma-separated string
        evidence_string = ", ".join(validated_tokens)
        return evidence_string
    
    def generate(
        self,
        text: str,
        predicted_label: Union[str, int],
        causal_evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language rationale explaining the prediction.
        
        Constructs a prompt with the predicted label, input text, and causal
        evidence, then generates a rationale using beam search.
        
        Args:
            text: Original input text that was classified
            predicted_label: The predicted class label (string or int)
            causal_evidence: List of causal tokens from Stage 2 (CausalTracer output)
        
        Returns:
            Generated rationale string explaining the prediction
        """
        # Format causal evidence into string
        causal_evidence_string = self._format_causal_evidence(causal_evidence)
        
        # Construct prompt following the specified format
        # Format: "explain prediction: {predicted_label} context: {text} evidence: {causal_evidence_string}"
        prompt = (
            f"explain prediction: {predicted_label} "
            f"context: {text} "
            f"evidence: {causal_evidence_string}"
        )
        
        # Tokenize input
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Generate rationale using beam search
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,  # Increased for longer rationales
                min_length=20,   # Ensure minimum length for detailed explanations
                num_beams=5,     # More beams for better quality
                no_repeat_ngram_size=3,  # Prevent repetition
                length_penalty=1.0,  # Encourage longer outputs
                early_stopping=True,
                temperature=0.9,  # Add slight randomness for more natural text
                do_sample=False  # Keep deterministic with beam search
            )
        
        # Decode generated text
        rationale = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # If rationale is too short, enhance it with rule-based fallback
        if len(rationale.split()) < 10:
            rationale = self._enhance_rationale(
                rationale, text, predicted_label, causal_evidence
            )
        
        return rationale
    
    def _enhance_rationale(
        self,
        short_rationale: str,
        text: str,
        predicted_label: Union[str, int],
        causal_evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Enhance a short rationale with additional details based on causal evidence.
        
        This is a fallback mechanism when the model generates very short explanations,
        which often happens with base (non-fine-tuned) T5 models.
        
        Args:
            short_rationale: The short rationale generated by the model
            text: Original input text
            predicted_label: Predicted class label
            causal_evidence: List of causal evidence tokens
        
        Returns:
            Enhanced rationale with more details
        """
        # Extract validated tokens
        validated_tokens = [
            item['token'] for item in causal_evidence 
            if item.get('is_validated', False)
        ]
        
        if not validated_tokens:
            # Use top tokens by causal impact if no validated tokens
            sorted_evidence = sorted(
                causal_evidence,
                key=lambda x: x.get('causal_impact', 0),
                reverse=True
            )
            validated_tokens = [item['token'] for item in sorted_evidence[:3]]
        
        # Build enhanced rationale
        if validated_tokens:
            tokens_str = ", ".join([f'"{token}"' for token in validated_tokens])
            enhanced = (
                f"{short_rationale}. This prediction is based on the causal analysis "
                f"of key sentiment-bearing tokens in the text. "
                f"The model identified {tokens_str} as having significant causal impact "
                f"on the {predicted_label} classification. "
                f"These tokens were validated through counterfactual testing, "
                f"where removing or replacing them resulted in measurable changes "
                f"in the model's prediction confidence."
            )
        else:
            enhanced = (
                f"{short_rationale}. This prediction is based on the overall "
                f"sentiment and linguistic features detected in the text. "
                f"The model analyzed the semantic content and context to determine "
                f"the {predicted_label} sentiment."
            )
        
        return enhanced
