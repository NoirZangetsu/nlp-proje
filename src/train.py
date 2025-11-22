"""
Training script for T5 rationale generator (Stage 3: Rationale Generation)

This script fine-tunes T5-small on the e-SNLI dataset to generate natural language
rationales explaining model predictions. The training is optimized for low VRAM
environments using gradient accumulation and mixed precision training.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import numpy as np

# Add parent directory to path to import config
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import config

# Try to import ROUGE evaluation library
ROUGE_AVAILABLE = False
USE_EVALUATE_LIB = False
rouge = None
rouge_scorer = None

try:
    import evaluate
    rouge = evaluate.load("rouge")
    ROUGE_AVAILABLE = True
    USE_EVALUATE_LIB = True
except ImportError:
    try:
        from rouge_score import rouge_scorer
        ROUGE_AVAILABLE = True
        USE_EVALUATE_LIB = False
    except ImportError:
        print("Warning: rouge_score and evaluate not available. ROUGE metrics will be skipped.")


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: T5Tokenizer,
    max_length: int = 512
) -> Dict[str, List]:
    """
    Preprocess e-SNLI examples for T5 training.
    
    Input format: "explain prediction: <premise> <hypothesis>"
    Target format: The explanation_1 field from the dataset.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: T5 tokenizer instance
        max_length: Maximum sequence length for truncation
        
    Returns:
        Dictionary with tokenized inputs and targets
    """
    # Get premises and hypotheses
    premises = examples["premise"]
    hypotheses = examples["hypothesis"]
    
    # Create input prompts
    inputs = [
        f"explain prediction: {premise} {hypothesis}"
        for premise, hypothesis in zip(premises, hypotheses)
    ]
    
    # Get target explanations
    targets = examples["explanation_1"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def compute_rouge_metrics(eval_pred, tokenizer: T5Tokenizer) -> Dict[str, float]:
    """
    Compute ROUGE scores for evaluation.
    
    Args:
        eval_pred: Tuple of predictions and labels from the trainer
        tokenizer: T5 tokenizer for decoding
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    if not ROUGE_AVAILABLE:
        return {}
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        predictions, 
        skip_special_tokens=True
    )
    
    # Decode labels (replace -100 with pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, 
        skip_special_tokens=True
    )
    
    # Use evaluate library if available (simpler)
    if USE_EVALUATE_LIB and rouge is not None:
        results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        # Rename keys to match expected format
        return {
            'rouge1_f1': results['rouge1'],
            'rouge2_f1': results['rouge2'],
            'rougeL_f1': results['rougeL']
        }
    
    # Fallback to rouge_score library
    if rouge_scorer is None:
        return {}
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], 
        use_stemmer=True
    )
    
    # Compute ROUGE scores
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_obj.score(label, pred)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[metric]['precision'].append(scores[metric].precision)
            rouge_scores[metric]['recall'].append(scores[metric].recall)
            rouge_scores[metric]['fmeasure'].append(scores[metric].fmeasure)
    
    # Average scores
    results = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        results[f'{metric}_precision'] = np.mean(rouge_scores[metric]['precision'])
        results[f'{metric}_recall'] = np.mean(rouge_scores[metric]['recall'])
        results[f'{metric}_f1'] = np.mean(rouge_scores[metric]['fmeasure'])
    
    return results


def load_and_prepare_data(
    tokenizer: T5Tokenizer,
    use_full_dataset: bool = False
):
    """
    Load e-SNLI dataset and prepare it for training.
    
    Args:
        tokenizer: T5 tokenizer instance
        use_full_dataset: If True, use 100% of training data; 
                         if False, use 10% (default)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print("Loading e-SNLI dataset...")
    dataset = load_dataset("esnli", trust_remote_code=True)
    
    # Select subset of training data (10% for demo, 100% for full training)
    if use_full_dataset:
        train_dataset = dataset["train"]
        print("Using 100% of training data")
    else:
        # Use 10% of training data
        train_size = len(dataset["train"])
        subset_size = int(0.1 * train_size)
        train_dataset = dataset["train"].select(range(subset_size))
        print(f"Using 10% of training data ({subset_size} examples)")
    
    # Use validation split for evaluation
    eval_dataset = dataset["validation"]
    print(f"Using {len(eval_dataset)} examples for evaluation")
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config.MAX_LENGTH),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config.MAX_LENGTH),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    return train_dataset, eval_dataset


def main():
    """
    Main training function.
    
    Sets up T5 model, loads data, configures training arguments,
    and runs the training loop.
    """
    print("=" * 60)
    print("T5 Rationale Generator Training")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./models/rationale_agent_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"\nLoading {config.GENERATOR_MODEL}...")
    tokenizer = T5Tokenizer.from_pretrained(config.GENERATOR_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(config.GENERATOR_MODEL)
    
    # Move model to device
    model.to(config.DEVICE)
    print(f"Model loaded on {config.DEVICE}")
    
    # Load and prepare data
    # Set use_full_dataset=True to use 100% of training data
    train_dataset, eval_dataset = load_and_prepare_data(
        tokenizer, 
        use_full_dataset=False
    )
    
    # Configure training arguments (optimized for low VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Small batch size for low VRAM
        gradient_accumulation_steps=8,   # Simulates batch size of 32
        fp16=True,                       # Mixed precision training
        num_train_epochs=3,
        save_total_limit=2,              # Keep only last 2 checkpoints
        evaluation_strategy="epoch",     # Evaluate after each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1_f1",
        greater_is_better=True,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # Disable wandb/tensorboard
        predict_with_generate=True,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_rouge_metrics(eval_pred, tokenizer)
    )
    
    # Train the model
    print("\nStarting training...")
    print(f"Training arguments:")
    print(f"  - Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Mixed precision (fp16): {training_args.fp16}")
    print(f"  - Number of epochs: {training_args.num_train_epochs}")
    print(f"  - Output directory: {output_dir}")
    print()
    
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training completed. Model saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
