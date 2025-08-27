"""
Inference utilities for the debiased model.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DebiasedModelInference:
    """Inference class for debiased language models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize inference model.
        
        Args:
            model_path: Path to the trained model
            device: Device to use for inference
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded debiased model from {model_path} on device {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing bias indicators.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return text.replace('[', '').replace(']', '').strip()
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
        """
        Make prediction on a single text.
        
        Args:
            text: Input text
            return_probabilities: Whether to return probabilities
            
        Returns:
            Prediction label or (label, probabilities) tuple
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
        
        if return_probabilities:
            return prediction, probabilities.cpu().numpy().flatten()
        return prediction
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, 
                     return_probabilities: bool = False) -> Union[List[int], Tuple[List[int], List[np.ndarray]]]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of predictions or (predictions, probabilities) tuple
        """
        # Preprocess texts
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        
        predictions = []
        probabilities = [] if return_probabilities else None
        
        # Process in batches
        for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Predicting"):
            batch_texts = cleaned_texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_probabilities = F.softmax(logits, dim=-1)
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy().tolist())
                
                if return_probabilities:
                    probabilities.extend(batch_probabilities.cpu().numpy())
        
        if return_probabilities:
            return predictions, probabilities
        return predictions
    
    def evaluate_bias_consistency(self, pro_sentences: List[str], 
                                anti_sentences: List[str]) -> Dict[str, float]:
        """
        Evaluate bias consistency between pro and anti-stereotyped sentences.
        
        Args:
            pro_sentences: List of pro-stereotyped sentences
            anti_sentences: List of anti-stereotyped sentences
            
        Returns:
            Dictionary of bias consistency metrics
        """
        assert len(pro_sentences) == len(anti_sentences), "Pro and anti sentences must have same length"
        
        logger.info(f"Evaluating bias consistency on {len(pro_sentences)} sentence pairs")
        
        # Get predictions with probabilities
        pro_predictions, pro_probs = self.predict_batch(pro_sentences, return_probabilities=True)
        anti_predictions, anti_probs = self.predict_batch(anti_sentences, return_probabilities=True)
        
        # Convert to numpy arrays
        pro_probs = np.array(pro_probs)
        anti_probs = np.array(anti_probs)
        
        # Calculate consistency metrics
        prediction_consistency = np.mean(np.array(pro_predictions) == np.array(anti_predictions))
        
        # Probability differences (using positive class probability)
        prob_differences = np.abs(pro_probs[:, 1] - anti_probs[:, 1])
        avg_prob_difference = np.mean(prob_differences)
        max_prob_difference = np.max(prob_differences)
        
        # Consistency score (1 - average absolute difference)
        consistency_score = 1 - avg_prob_difference
        
        # Count of perfectly consistent pairs (difference < 0.01)
        perfect_consistency_count = np.sum(prob_differences < 0.01)
        perfect_consistency_ratio = perfect_consistency_count / len(pro_sentences)
        
        # Bias magnitude
        bias_magnitude = np.mean(prob_differences)
        
        metrics = {
            'prediction_consistency': prediction_consistency,
            'avg_probability_difference': avg_prob_difference,
            'max_probability_difference': max_prob_difference,
            'consistency_score': consistency_score,
            'perfect_consistency_ratio': perfect_consistency_ratio,
            'bias_magnitude': bias_magnitude,
            'total_pairs': len(pro_sentences)
        }
        
        logger.info(f"Bias consistency metrics: {metrics}")
        return metrics
    
    def analyze_sentence_pair(self, pro_sentence: str, anti_sentence: str) -> Dict[str, any]:
        """
        Detailed analysis of a single sentence pair.
        
        Args:
            pro_sentence: Pro-stereotyped sentence
            anti_sentence: Anti-stereotyped sentence
            
        Returns:
            Dictionary with detailed analysis
        """
        # Get predictions with probabilities
        pro_pred, pro_probs = self.predict_single(pro_sentence, return_probabilities=True)
        anti_pred, anti_probs = self.predict_single(anti_sentence, return_probabilities=True)
        
        # Calculate differences
        prob_difference = abs(pro_probs[1] - anti_probs[1])
        prediction_consistent = pro_pred == anti_pred
        
        analysis = {
            'pro_sentence': pro_sentence,
            'anti_sentence': anti_sentence,
            'pro_prediction': int(pro_pred),
            'anti_prediction': int(anti_pred),
            'pro_probabilities': pro_probs.tolist(),
            'anti_probabilities': anti_probs.tolist(),
            'probability_difference': float(prob_difference),
            'prediction_consistent': prediction_consistent,
            'consistency_score': float(1 - prob_difference),
            'bias_direction': 'pro' if pro_probs[1] > anti_probs[1] else 'anti'
        }
        
        return analysis
    
    def batch_analyze_pairs(self, sentence_pairs: List[Tuple[str, str]], 
                          save_path: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Analyze multiple sentence pairs.
        
        Args:
            sentence_pairs: List of (pro, anti) sentence tuples
            save_path: Optional path to save analysis results
            
        Returns:
            List of analysis dictionaries
        """
        logger.info(f"Analyzing {len(sentence_pairs)} sentence pairs")
        
        results = []
        for i, (pro, anti) in enumerate(tqdm(sentence_pairs, desc="Analyzing pairs")):
            analysis = self.analyze_sentence_pair(pro, anti)
            analysis['pair_id'] = i
            results.append(analysis)
        
        # Calculate summary statistics
        consistency_scores = [r['consistency_score'] for r in results]
        prob_differences = [r['probability_difference'] for r in results]
        prediction_consistencies = [r['prediction_consistent'] for r in results]
        
        summary = {
            'total_pairs': len(sentence_pairs),
            'avg_consistency_score': np.mean(consistency_scores),
            'std_consistency_score': np.std(consistency_scores),
            'avg_probability_difference': np.mean(prob_differences),
            'prediction_consistency_rate': np.mean(prediction_consistencies),
            'perfect_consistency_count': sum(1 for d in prob_differences if d < 0.01)
        }
        
        analysis_result = {
            'summary': summary,
            'detailed_results': results
        }
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis results saved to {save_path}")
        
        logger.info(f"Analysis summary: {summary}")
        return results


def load_test_data(data_path: str) -> List[Tuple[str, str]]:
    """
    Load test data from JSON file.
    
    Args:
        data_path: Path to test data file
        
    Returns:
        List of (pro, anti) sentence pairs
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = [(item['pro'], item['anti']) for item in data]
    logger.info(f"Loaded {len(pairs)} sentence pairs from {data_path}")
    return pairs


if __name__ == "__main__":
    # Example usage
    model_path = "outputs/models/best_model"  # Update with actual path
    
    if os.path.exists(model_path):
        # Initialize inference
        inference = DebiasedModelInference(model_path)
        
        # Test single prediction
        test_sentence = "The developer argued with the designer because he did not like the design."
        prediction = inference.predict_single(test_sentence, return_probabilities=True)
        logger.info(f"Test prediction: {prediction}")
        
        # Test bias consistency if test data is available
        test_data_path = "data/processed/splits/test.json"
        if os.path.exists(test_data_path):
            test_pairs = load_test_data(test_data_path)
            
            # Take a sample for testing
            sample_pairs = test_pairs[:10]
            pro_sentences = [pair[0] for pair in sample_pairs]
            anti_sentences = [pair[1] for pair in sample_pairs]
            
            # Evaluate bias consistency
            metrics = inference.evaluate_bias_consistency(pro_sentences, anti_sentences)
            logger.info(f"Bias consistency metrics: {metrics}")
            
            # Detailed analysis
            analysis_results = inference.batch_analyze_pairs(
                sample_pairs, 
                "outputs/reports/bias_analysis.json"
            )
            
            logger.info("Inference testing completed!")
        else:
            logger.warning(f"Test data not found at {test_data_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Please train a model first.")