"""
Adaptive Model Fusion

This module implements a novel approach for dynamically fusing outputs from multiple models
based on input characteristics and historical performance. The approach uses a combination of:

1. Input Feature Analysis: Extracts features from inputs to determine which models are likely to perform best
2. Performance History: Tracks model performance on similar inputs to adjust fusion weights
3. Confidence-based Weighting: Uses model confidence scores to weight their contributions
4. Adaptive Learning: Continuously updates fusion weights based on feedback

This approach goes beyond simple ensemble methods by dynamically adjusting the fusion strategy
for each input, leading to better performance across diverse inputs.
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/adaptive_fusion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class AdaptiveModelFusion:
    """
    A novel approach for dynamically fusing outputs from multiple models based on
    input characteristics and historical performance.
    """
    
    def __init__(self, model_manager, history_file="results/fusion_history.json", learning_rate=0.1):
        """
        Initialize the Adaptive Model Fusion system.
        
        Args:
            model_manager: The model manager instance for loading models
            history_file: File to store performance history
            learning_rate: Rate at which to update weights based on feedback
        """
        self.model_manager = model_manager
        self.history_file = history_file
        self.learning_rate = learning_rate
        
        # Initialize model weights for each task
        self.model_weights = {
            "summarization": {"qwen": 0.4, "opt": 0.3, "llama": 0.3},
            "qa": {"qwen": 0.4, "opt": 0.3, "llama": 0.3},
            "paraphrase": {"qwen": 0.3, "opt": 0.3, "llama": 0.4}
        }
        
        # Load performance history if available
        self.performance_history = self._load_history()
        
        # Initialize feature extractors
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.stop_words = set(stopwords.words('english'))
        
        # Cache for similar inputs
        self.similarity_cache = {}
        
    def _load_history(self) -> Dict:
        """Load performance history from file if it exists"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history file: {e}")
                return self._initialize_history()
        else:
            return self._initialize_history()
    
    def _initialize_history(self) -> Dict:
        """Initialize an empty performance history"""
        return {
            "summarization": {"qwen": [], "opt": [], "llama": []},
            "qa": {"qwen": [], "opt": [], "llama": []},
            "paraphrase": {"qwen": [], "opt": [], "llama": []}
        }
    
    def _save_history(self):
        """Save performance history to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f)
        except Exception as e:
            logger.error(f"Error saving history file: {e}")
    
    def _extract_features(self, input_text: str) -> np.ndarray:
        """
        Extract features from input text for similarity comparison.
        
        Args:
            input_text: The input text to extract features from
            
        Returns:
            Feature vector
        """
        if isinstance(input_text, dict):
            # For QA task, combine context and question
            if "context" in input_text and "question" in input_text:
                text = input_text["context"] + " " + input_text["question"]
            else:
                text = str(input_text)
        else:
            text = input_text
            
        # Tokenize and remove stop words
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        # Extract features using TF-IDF
        try:
            # If vectorizer is not fitted yet, fit it
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit([" ".join(filtered_tokens)])
                
            features = self.vectorizer.transform([" ".join(filtered_tokens)])
            return features.toarray()[0]
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return a default feature vector
            return np.ones(100) / 100
    
    def _find_similar_inputs(self, task: str, features: np.ndarray, threshold=0.7) -> List[Dict]:
        """
        Find similar inputs in the performance history.
        
        Args:
            task: The task type
            features: Feature vector of the current input
            threshold: Similarity threshold
            
        Returns:
            List of similar inputs with their performance data
        """
        similar_inputs = []
        
        # Check cache first
        cache_key = f"{task}_{hash(str(features.tolist()))}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Check each model's history
        for model_name in self.model_weights[task].keys():
            for entry in self.performance_history[task][model_name]:
                if "features" in entry:
                    entry_features = np.array(entry["features"])
                    similarity = cosine_similarity([features], [entry_features])[0][0]
                    
                    if similarity > threshold:
                        similar_inputs.append({
                            "model": model_name,
                            "score": entry["score"],
                            "similarity": similarity
                        })
        
        # Cache the result
        self.similarity_cache[cache_key] = similar_inputs
        return similar_inputs
    
    def _adjust_weights(self, task: str, similar_inputs: List[Dict]):
        """
        Adjust model weights based on similar inputs' performance.
        
        Args:
            task: The task type
            similar_inputs: List of similar inputs with their performance data
        """
        if not similar_inputs:
            return
        
        # Calculate average score for each model
        model_scores = defaultdict(list)
        for entry in similar_inputs:
            model_scores[entry["model"]].append(entry["score"] * entry["similarity"])
        
        avg_scores = {}
        for model, scores in model_scores.items():
            avg_scores[model] = sum(scores) / len(scores) if scores else 0
        
        # Normalize scores to get weights
        total_score = sum(avg_scores.values())
        if total_score > 0:
            new_weights = {model: score / total_score for model, score in avg_scores.items()}
            
            # Update weights with learning rate
            for model in self.model_weights[task]:
                if model in new_weights:
                    self.model_weights[task][model] = (1 - self.learning_rate) * self.model_weights[task][model] + \
                                                     self.learning_rate * new_weights[model]
            
            # Normalize weights
            total_weight = sum(self.model_weights[task].values())
            for model in self.model_weights[task]:
                self.model_weights[task][model] /= total_weight
    
    def _update_history(self, task: str, model: str, input_features: List[float], score: float):
        """
        Update performance history with new data.
        
        Args:
            task: The task type
            model: The model name
            input_features: Features of the input
            score: Performance score
        """
        # Add new entry
        self.performance_history[task][model].append({
            "features": input_features.tolist(),
            "score": score,
            "timestamp": time.time()
        })
        
        # Limit history size to prevent it from growing too large
        max_history = 100
        if len(self.performance_history[task][model]) > max_history:
            self.performance_history[task][model] = self.performance_history[task][model][-max_history:]
        
        # Save updated history
        self._save_history()
    
    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float]:
        """
        Generate output using adaptive model fusion.
        
        Args:
            task: The task type
            input_text: The input text
            prompt_template: The prompt template
            
        Returns:
            Tuple of (fused_output, inference_time)
        """
        start_time = time.time()
        
        # Extract features from input
        features = self._extract_features(input_text)
        
        # Find similar inputs in history
        similar_inputs = self._find_similar_inputs(task, features)
        
        # Adjust weights based on similar inputs
        self._adjust_weights(task, similar_inputs)
        
        logger.info(f"Adaptive fusion weights for {task}: {self.model_weights[task]}")
        
        # Generate outputs from each model
        model_outputs = {}
        model_confidences = {}
        
        for model_name, weight in self.model_weights[task].items():
            if weight > 0.1:  # Only use models with significant weight
                try:
                    # Load model
                    model, tokenizer = self.model_manager.load_model(model_name, task)
                    
                    # Prepare input based on task
                    if task == "qa" and isinstance(input_text, dict):
                        prompt = prompt_template.format(context=input_text['context'], question=input_text['question'])
                    elif task == "summarization":
                        prompt = prompt_template.format(article=input_text)
                    elif task == "paraphrase":
                        prompt = prompt_template.format(input_question=input_text)
                    else:
                        prompt = prompt_template.format(**{k: input_text for k in ["article", "context", "question", "input_question"]})
                    
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
                    
                    # Generate with temperature sampling to get confidence
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            max_new_tokens=100,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    # Decode
                    input_length = inputs.input_ids.shape[1]
                    generated_ids = outputs.sequences[0, input_length:]
                    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    # Calculate confidence score (average log probability)
                    if hasattr(outputs, 'scores') and outputs.scores:
                        log_probs = []
                        for i, logits in enumerate(outputs.scores):
                            # Get the predicted token's logit
                            token_id = outputs.sequences[0, input_length + i].item()
                            token_logit = logits[0, token_id].item()
                            log_probs.append(token_logit)
                        
                        # Average log probability as confidence
                        confidence = np.mean(log_probs) if log_probs else 0
                    else:
                        confidence = 0
                    
                    # Store output and confidence
                    model_outputs[model_name] = output_text
                    model_confidences[model_name] = confidence
                    
                    # Unload model
                    self.model_manager.unload_model(model_name, task)
                    
                except Exception as e:
                    logger.error(f"Error generating with {model_name} for {task}: {e}")
                    model_outputs[model_name] = ""
                    model_confidences[model_name] = 0
        
        # Fuse outputs based on weights and confidences
        if not model_outputs:
            logger.error(f"No model outputs generated for {task}")
            return "", time.time() - start_time
        
        # Normalize confidences
        total_confidence = sum(model_confidences.values())
        if total_confidence > 0:
            normalized_confidences = {model: conf / total_confidence for model, conf in model_confidences.items()}
        else:
            normalized_confidences = {model: 1.0 / len(model_confidences) for model in model_confidences}
        
        # Combine weights and confidences
        combined_weights = {}
        for model in model_outputs:
            combined_weights[model] = 0.7 * self.model_weights[task][model] + 0.3 * normalized_confidences[model]
        
        # Normalize combined weights
        total_combined_weight = sum(combined_weights.values())
        if total_combined_weight > 0:
            combined_weights = {model: weight / total_combined_weight for model, weight in combined_weights.items()}
        
        # Select the model with the highest combined weight
        best_model = max(combined_weights.items(), key=lambda x: x[1])[0]
        fused_output = model_outputs[best_model]
        
        # Update performance history (placeholder score until feedback)
        self._update_history(task, best_model, features, 0.5)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        return fused_output, inference_time
    
    def update_feedback(self, task: str, input_text: str, model: str, score: float):
        """
        Update the system with feedback on the quality of a generated output.
        
        Args:
            task: The task type
            input_text: The input text
            model: The model that generated the output
            score: Quality score (0-1)
        """
        features = self._extract_features(input_text)
        self._update_history(task, model, features, score)
        
        # Clear similarity cache for this input
        cache_key = f"{task}_{hash(str(features.tolist()))}"
        if cache_key in self.similarity_cache:
            del self.similarity_cache[cache_key]
