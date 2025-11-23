"""
Streamlit dashboard for Natural Language Rationale Generation Agent

This app visualizes the complete XAI pipeline:
1. BERT prediction with confidence
2. Saliency visualization (Integrated Gradients)
3. Causal analysis (Counterfactuals)
4. Natural language rationale generation
"""
import streamlit as st
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Any
import re

from src.interpretation import ModelInterpreter
from src.causal import CausalTracer
from src.generation import RationaleGenerator
import config


# Page configuration
st.set_page_config(
    page_title="XAI Rationale Generation Agent",
    page_icon="🤖",
    layout="wide"
)


def highlight_text_with_attributions(
    text: str,
    attributions: List[Dict[str, float]],
    tokenizer
) -> str:
    """
    Generate HTML to highlight text based on Integrated Gradients scores.
    
    Colors words red (negative impact) or green (positive impact) based on
    their attribution scores. Intensity of color corresponds to magnitude.
    
    Args:
        text: Original input text
        attributions: List of dictionaries with 'token' and 'score' keys
        tokenizer: Tokenizer used to tokenize the text
        
    Returns:
        HTML string with highlighted text
    """
    if not attributions:
        # Return plain text if no attributions
        escaped_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return f'<div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; color: #212529;">{escaped_text}</div>'
    
    # Tokenize the text to get token positions
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_offsets_mapping=True
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    offsets = encoded["offset_mapping"][0]
    
    # Create a mapping from token to score
    # Handle subword tokens (e.g., "##ing" maps to "ing")
    token_scores = {}
    for attr in attributions:
        token = attr['token']
        score = attr['score']
        # Store both the full token and cleaned version
        token_scores[token] = score
        # Handle subword tokens
        if token.startswith('##'):
            token_scores[token[2:]] = score
    
    # Normalize scores for color intensity (0-1 range)
    scores = [attr['score'] for attr in attributions]
    max_score = max(abs(s) for s in scores) if scores else 1.0
    
    # Build highlighted HTML by processing text with offsets
    highlighted_parts = []
    last_end = 0
    
    for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        # Skip special tokens and padding
        if token in [tokenizer.cls_token, tokenizer.sep_token, 
                    tokenizer.pad_token, '[CLS]', '[SEP]', '[PAD]']:
            continue
        
        # Handle offset mapping (None means special token)
        if start is None or end is None:
            continue
        
        # Get score for this token (try exact match first, then cleaned)
        score = token_scores.get(token, 0.0)
        if score == 0.0 and token.startswith('##'):
            score = token_scores.get(token[2:], 0.0)
        
        # Normalize score to 0-1 range for color intensity
        normalized_score = abs(score) / max_score if max_score > 0 else 0.0
        normalized_score = min(normalized_score, 1.0)
        
        # Determine color: green for positive, red for negative
        if score > 0:
            # Green scale (positive impact) - darker green for higher scores
            alpha = normalized_score
            bg_color = f"rgba(144, 238, 144, {0.3 + 0.5 * alpha})"  # Light green with varying opacity
            border_color = f"rgba(0, 128, 0, {alpha})"
        elif score < 0:
            # Red scale (negative impact) - darker red for higher scores
            alpha = normalized_score
            bg_color = f"rgba(255, 182, 193, {0.3 + 0.5 * alpha})"  # Light red with varying opacity
            border_color = f"rgba(128, 0, 0, {alpha})"
        else:
            # Neutral (gray)
            bg_color = "rgba(240, 240, 240, 0.5)"
            border_color = "rgba(128, 128, 128, 0.3)"
        
        # Add any text between last token and current token
        if start > last_end:
            gap_text = text[last_end:start]
            gap_text = gap_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            highlighted_parts.append(f'<span style="color: #212529;">{gap_text}</span>')
        
        # Get the actual text span for this token
        if start < len(text) and end <= len(text):
            token_text = text[start:end]
            # Escape HTML special characters
            token_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Create highlighted span
            highlighted_parts.append(
                f'<span style="background-color: {bg_color}; '
                f'border-left: 3px solid {border_color}; '
                f'padding: 2px 4px; margin: 1px; '
                f'border-radius: 3px; display: inline-block; '
                f'color: #212529;">{token_text}</span>'
            )
            
            last_end = end
    
    # Add any remaining text
    if last_end < len(text):
        remaining_text = text[last_end:]
        remaining_text = remaining_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        highlighted_parts.append(f'<span style="color: #212529;">{remaining_text}</span>')
    
    # Join all parts
    highlighted_html = ''.join(highlighted_parts)
    
    # Wrap in a container div
    return f'<div style="line-height: 2.0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; color: #212529;">{highlighted_html}</div>'


def get_class_label(class_idx: int) -> str:
    """
    Get human-readable class label for SST-2 predictions.
    
    Args:
        class_idx: Class index (0 or 1)
        
    Returns:
        Class label string
    """
    labels = {0: "Negative", 1: "Positive"}
    return labels.get(class_idx, f"Class {class_idx}")


def main():
    """Main Streamlit app function."""
    
    # Header
    st.title("🤖 CSE 655 Project: XAI Rationale Generation Agent")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Model selection (placeholder for future expansion)
        st.subheader("Model Settings")
        model_name = st.selectbox(
            "Select Model",
            ["BERT-base-uncased-SST-2"],
            index=0
        )
        
        # Threshold settings
        st.subheader("Causal Analysis Settings")
        causal_threshold = st.slider(
            "Causal Impact Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Minimum causal impact score to validate a token"
        )
        
        top_n_tokens = st.slider(
            "Top N Tokens to Analyze",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of top tokens to test for causal impact"
        )
        
        st.markdown("---")
        st.info("💡 Enter text below and click 'Explain Decision' to see the XAI pipeline in action.")
    
    # Main area
    st.header("📝 Input")
    input_text = st.text_area(
        "Enter movie review or premise",
        height=150,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot was engaging."
    )
    
    explain_button = st.button("🔍 Explain Decision", type="primary", width="stretch")
    
    if explain_button and input_text.strip():
        try:
            # Initialize components
            with st.spinner("Initializing models..."):
                interpreter = ModelInterpreter()
                causal_tracer = CausalTracer(interpreter)
                rationale_generator = RationaleGenerator()
            
            # Step 1: BERT Prediction
            st.markdown("---")
            st.header("📊 Step 1: BERT Prediction")
            
            with st.spinner("Computing prediction..."):
                probs, predicted_class = interpreter.predict(input_text)
                predicted_label = get_class_label(predicted_class)
                confidence = probs[predicted_class].item()
            
            # Display prediction with confidence bar
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Predicted Label", predicted_label)
                st.metric("Confidence", f"{confidence:.2%}")
            
            with col2:
                # Confidence bar
                st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                
                # Probability distribution
                prob_dict = {
                    "Negative": probs[0].item(),
                    "Positive": probs[1].item()
                }
                st.bar_chart(prob_dict)
            
            # Step 2: Saliency (Integrated Gradients)
            st.markdown("---")
            st.header("🎯 Step 2: Saliency Analysis (Integrated Gradients)")
            
            with st.spinner("Computing integrated gradients..."):
                attributions = interpreter.compute_integrated_gradients(input_text)
            
            st.subheader("Token Importance Visualization")
            st.markdown(
                "**Color coding:** 🟢 Green = Positive impact, 🔴 Red = Negative impact"
            )
            
            # Generate highlighted HTML
            highlighted_html = highlight_text_with_attributions(
                input_text,
                attributions,
                interpreter.tokenizer
            )
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # Show top tokens
            if attributions:
                st.subheader("Top Important Tokens")
                top_attributions = sorted(
                    attributions,
                    key=lambda x: abs(x['score']),
                    reverse=True
                )[:10]
                
                top_df = pd.DataFrame({
                    'Token': [a['token'] for a in top_attributions],
                    'Score': [a['score'] for a in top_attributions]
                })
                st.dataframe(top_df, width="stretch", hide_index=True)
            
            # Step 3: Causal Analysis
            st.markdown("---")
            st.header("🔬 Step 3: Causal Analysis (Counterfactuals)")
            
            with st.spinner("Generating counterfactuals and computing causal impact..."):
                causal_results = causal_tracer.generate_counterfactuals(
                    input_text,
                    top_n_tokens=top_n_tokens
                )
            
            if causal_results:
                st.subheader("Causal Impact Analysis")
                st.markdown(
                    f"**Causal Impact Threshold:** {causal_threshold:.2f}\n\n"
                    "Tokens with causal impact above the threshold are highlighted."
                )
                
                # Create DataFrame
                causal_df = pd.DataFrame({
                    'Token': [r['token'] for r in causal_results],
                    'Saliency': [f"{r['saliency']:.4f}" for r in causal_results],
                    'Causal Impact': [f"{r['causal_impact']:.4f}" for r in causal_results],
                    'Is Effective?': ['✅ Yes' if r['is_validated'] else '❌ No' 
                                     for r in causal_results]
                })
                
                # Style the DataFrame to highlight effective rows
                def highlight_effective(row):
                    is_effective = row['Is Effective?'] == '✅ Yes'
                    return ['background-color: #d4edda' if is_effective else '' 
                           for _ in row]
                
                styled_df = causal_df.style.apply(highlight_effective, axis=1)
                st.dataframe(styled_df, width="stretch", hide_index=True)
            else:
                st.warning("No causal evidence found. Try a different input text.")
            
            # Step 4: Rationale Generation
            st.markdown("---")
            st.header("💬 Step 4: Natural Language Rationale")
            
            with st.spinner("Generating rationale..."):
                rationale = rationale_generator.generate(
                    text=input_text,
                    predicted_label=predicted_label,
                    causal_evidence=causal_results if causal_results else []
                )
            
            # Display rationale in styled blockquote
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    font-style: italic;
                    color: #212529;
                ">
                    <strong style="color: #212529;">🤖 Agent Rationale:</strong><br>
                    <span style="color: #212529;">{rationale}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except torch.cuda.OutOfMemoryError as e:
            st.error("❌ **CUDA Out of Memory Error**")
            st.warning(
                "The model requires more GPU memory than available. "
                "Try:\n"
                "- Using a shorter input text\n"
                "- Restarting the app to clear GPU cache\n"
                "- Using CPU mode (if CUDA is not essential)"
            )
            st.exception(e)
            
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
            st.exception(e)
    
    elif explain_button and not input_text.strip():
        st.warning("⚠️ Please enter some text before clicking 'Explain Decision'.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'>"
        "XAI Rationale Generation Agent | Stage 1: Saliency → Stage 2: Causality → Stage 3: Generation"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
