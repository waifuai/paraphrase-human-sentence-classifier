#!/usr/bin/env python3
"""
Model definition for the Human/Paraphrase Sentence Classifier using Hugging Face Transformers.
"""

from transformers import AutoModelForSequenceClassification, PreTrainedModel

def load_model(model_name: str = "distilbert-base-uncased", num_labels: int = 2) -> PreTrainedModel:
    """
    Loads a pre-trained Transformer model for sequence classification.

    Args:
        model_name: The name or path of the pre-trained model from Hugging Face Hub.
        num_labels: The number of output labels for the classification task.

    Returns:
        A pre-trained Transformer model configured for sequence classification.
    """
    print(f"Loading model: {model_name} for {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print("Model loaded successfully.")
    return model

# Example usage (for testing purposes)
if __name__ == "__main__":
    try:
        # Load a small model for quick testing
        test_model_name = "prajjwal1/bert-tiny" # A smaller BERT model
        model = load_model(model_name=test_model_name, num_labels=2)
        print(f"\nLoaded model config:\n{model.config}")

        # Test with a different number of labels
        model_3_labels = load_model(model_name=test_model_name, num_labels=3)
        print(f"\nLoaded model config (3 labels):\n{model_3_labels.config}")

    except Exception as e:
        print(f"Error during model loading test: {e}")
        print("Ensure you have an internet connection and the necessary libraries installed.")