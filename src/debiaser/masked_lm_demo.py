from transformers import AutoTokenizer, DistilBertForMaskedLM
import torch
import torch.nn.functional as F


# Detect and set the best available device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Metal Performance Shaders for Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA CUDA
    else:
        return torch.device("cpu")  # CPU fallback


device = get_device()
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model = model.to(device)  # Move model to the selected device


def predict_masked_word(sentence, top_k=5):
    """
    Predict the top k words for a sentence containing <mask> token.

    Args:
        sentence (str): Input sentence containing exactly one <mask> token
        top_k (int): Number of top predictions to return (default: 5)

    Returns:
        list: List of tuples containing (word, probability)
    """
    # Replace <mask> with the actual mask token
    sentence = sentence.replace(
        "<mask>", tokenizer.mask_token
    )  # tokenizer.mask_token = [MASK]

    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device

    # Find the position of the mask token
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(
        as_tuple=True
    )[0]

    if len(mask_token_index) == 0:
        raise ValueError("No <mask> token found in the sentence")
    if len(mask_token_index) > 1:
        raise ValueError(
            "Multiple <mask> tokens found. Please provide exactly one <mask> token."
        )

    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get logits for the mask position
    mask_logits = logits[0, mask_token_index[0]]

    # Apply softmax to get probabilities
    probabilities = F.softmax(mask_logits, dim=-1)

    # Get top k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

    # Decode tokens and create results
    results = []
    for i in range(top_k):
        token = tokenizer.decode(top_k_indices[i].item())
        prob = top_k_probs[i].item()
        results.append((token, prob))

    return results


# Example usage
if __name__ == "__main__":
    # Test with a sample sentence
    # test_sentence = "The doctor was a excellent surgeon. <mask> really saved my life."
    # test_sentence = "The doctor was a bad surgeon. <mask> barely saved my life."
    test_sentence = "The Food was really good. <mask> must be talented."
    print(f"Input sentence: {test_sentence}")
    print("\nTop 5 predictions:")
    print("-" * 30)

    predictions = predict_masked_word(test_sentence, top_k=5)

    for i, (word, prob) in enumerate(predictions, 1):
        print(f"{i}. {word:<15} ({prob:.4f})")

    print("\n" + "=" * 50)
