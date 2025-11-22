import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel with data structures to store n-gram counts.
        """
        # Dictionary to store trigram counts: {(w1, w2): {w3: count, ...}, ...}
        self.ngrams = defaultdict(lambda: defaultdict(int))
        # Dictionary to store bigram counts for smoothing
        self.bigrams = defaultdict(int)
        # Vocabulary to track all unique words
        self.vocab = set()
        # Special tokens
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.unknown_token = "<unk>"
        # For random number generation
        self.random = random.Random(42)  # Fixed seed for reproducibility

    def _clean_text(self, text: str) -> str:
        """
        Cleans the input text by normalizing punctuation, fixing contractions, 
        and handling special characters.
        
        Args:
            text: The input text to clean.
            
        Returns:
            The cleaned text with normalized punctuation and spacing.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle common contractions and possessives
        text = re.sub(r"(\w+)'s\b", r"\1 's", text)  # Handle 's possessives
        text = re.sub(r"(\w+)n't\b", r"\1 n't", text)  # Handle n't contractions
        text = re.sub(r"(\w+)\'ll\b", r"\1 'll", text)  # Handle 'll
        text = re.sub(r"(\w+)\'re\b", r"\1 're", text)  # Handle 're
        text = re.sub(r"(\w+)\'ve\b", r"\1 've", text)  # Handle 've
        text = re.sub(r"(\w+)\'d\b", r"\1 'd", text)    # Handle 'd
        text = re.sub(r"(\w+)\'m\b", r"\1 'm", text)    # Handle 'm
        
        # Replace newlines and tabs with spaces
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Handle punctuation spacing
        text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)  # Add space after punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
        
        # Remove special characters but keep basic punctuation and apostrophes
        text = re.sub(r"[^a-z0-9\s.,!?']", ' ', text)
        
        # Fix multiple spaces and strip
        text = ' '.join(text.split())
        
        # Remove Project Gutenberg header/footer if present
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "***START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG"
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "***END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG"
        ]
        
        # Find and remove header
        for marker in start_markers:
            start_idx = text.lower().find(marker.lower())
            if start_idx != -1:
                text = text[start_idx + len(marker):]
                break
        
        # Find and remove footer
        for marker in end_markers:
            end_idx = text.lower().find(marker.lower())
            if end_idx != -1:
                text = text[:end_idx]
                break
        
        # Remove chapter headings and section breaks
        text = re.sub(r'(?i)chapter\s+[ivxlcdm]+\s*\n', '\n', text)
        text = re.sub(r'\*{2,}.*?\*{2,}', ' ', text, flags=re.DOTALL)
        
        # Handle common contractions and special characters
        text = text.replace('’', "'").replace('‘', "'").replace('”', '"')
        text = text.replace('“', '"').replace('—', ' -- ').replace('–', '-')
        
        # Replace various whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any remaining non-word characters except basic punctuation and spaces
        text = re.sub(r'[^\w\s\.,!?\-\'\"]', ' ', text)
        
        # Normalize spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])(?=[^ \n])', r'\1 ', text)  # Add space after punctuation if missing
        
        # Handle multiple sentences on one line
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1 \2', text)
        
        # Final cleanup
        text = text.lower().strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Split on word boundaries, keeping contractions together
        tokens = []
        for token in re.findall(r"\b[\w']+\b|[.!?]+", text):
            # Handle common contractions and possessives
            if "'" in token.lower() and token.lower() not in ["i'm", "i'll", "i've", "i'd", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "wouldn't", "couldn't", "shouldn't", "let's", "that's", "who's", "what's", "where's", "when's", "why's", "how's"]:
                # Split possessives (e.g., "alice's" -> "alice", "'s")
                parts = token.split("'")
                tokens.append(parts[0].lower())
                if len(parts) > 1 and parts[1]:
                    tokens.append("'" + parts[1].lower())
            else:
                tokens.append(token.lower())
        
        # Remove empty tokens and normalize
        tokens = [token for token in tokens if token.strip()]
        return tokens

    def _add_start_end_tokens(self, tokens: List[str]) -> List[str]:
        """
        Adds start and end tokens to the token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with start and end tokens
        """
        return [self.start_token, self.start_token] + tokens + [self.end_token]

    def _replace_rare_words(self, tokens: List[str], min_count: int = 1) -> List[str]:
        """
        Replaces rare words with the unknown token.
        
        Args:
            tokens: List of tokens
            min_count: Minimum count for a word to be kept in vocabulary
            
        Returns:
            List of tokens with rare words replaced
        """
        # Count word frequencies
        word_counts = defaultdict(int)
        for token in tokens:
            if token not in [self.start_token, self.end_token]:
                word_counts[token] += 1
        
        # Replace rare words
        processed_tokens = []
        for token in tokens:
            if token in [self.start_token, self.end_token]:
                processed_tokens.append(token)
            elif word_counts[token] >= min_count:
                processed_tokens.append(token)
                self.vocab.add(token)
            else:
                processed_tokens.append(self.unknown_token)
                self.vocab.add(self.unknown_token)
        
        return processed_tokens

    def fit(self, text: str, min_word_count: int = 1):
        """
        Trains the trigram model on the given text.

        Args:
            text: The text to train the model on.
            min_word_count: Minimum count for a word to be kept in vocabulary
        """
        if not text.strip():
            return
            
        # Clean and tokenize the text
        cleaned_text = self._clean_text(text)
        tokens = self._tokenize(cleaned_text)
        
        # Add start/end tokens and handle rare words
        tokens = self._add_start_end_tokens(tokens)
        tokens = self._replace_rare_words(tokens, min_word_count)
        
        # Count n-grams
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            
            # Update trigram counts
            self.ngrams[(w1, w2)][w3] += 1
            
            # Update bigram counts for smoothing
            self.bigrams[(w1, w2)] += 1

    def _get_next_word(self, w1: str, w2: str, temperature: float = 1.0) -> str:
        """
        Gets the next word based on the previous two words using probability sampling.
        
        Args:
            w1: First word in the context
            w2: Second word in the context
            temperature: Controls randomness in the generation. 
                       Higher values (e.g., 1.0) make output more random, 
                       lower values (e.g., 0.1) make it more deterministic.
            
        Returns:
            Next word
        """
        # If we have this context, use it
        if (w1, w2) in self.ngrams and self.ngrams[(w1, w2)]:
            next_words = list(self.ngrams[(w1, w2)].items())
            
            # Calculate probabilities with temperature
            total = sum(count for word, count in next_words)
            if total == 0:
                next_word = self.random.choice(list(self.vocab)) if self.vocab else self.end_token
                return next_word
                
            # Apply temperature to logits
            if temperature <= 0:
                # If temperature is 0, always pick the most likely word
                return max(next_words, key=lambda x: x[1])[0]
                
            # Apply temperature scaling
            probs = [(word, (count/total) ** (1.0/temperature)) for word, count in next_words]
            
            # Normalize probabilities
            prob_sum = sum(prob for _, prob in probs)
            if prob_sum > 0:
                probs = [(word, prob/prob_sum) for word, prob in probs]
                
                # Sample based on probability
                r = self.random.random()
                cumulative = 0
                for word, prob in probs:
                    cumulative += prob
                    if r < cumulative or cumulative > 0.9999:  # Handle floating point errors
                        return word
        
        # Fallback 1: Try to find a similar context by using just the last word
        if w2 in self.vocab:
            possible_contexts = [k for k in self.ngrams.keys() if k[1] == w2]
            if possible_contexts:
                # Choose a random context that ends with w2
                context = self.random.choice(possible_contexts)
                next_words = list(self.ngrams[context].items())
                if next_words:
                    return max(next_words, key=lambda x: x[1])[0]
        
        # Fallback 2: Return a random word from the vocabulary
        if self.vocab:
            return self.random.choice(list(self.vocab))
            
        # Last resort: return end token
        return self.end_token

    def generate(self, max_length: int = 100, temperature: float = 0.8, seed: int = None) -> str:
        """
        Generates new text using the trained trigram model with improved coherence.

        Args:
            max_length: The maximum number of words to generate.
            temperature: Controls randomness in generation (0.1-1.0).
                       Lower values make output more predictable.
            seed: Optional random seed for reproducibility.

        Returns:
            The generated text with proper formatting.
        """
        if not self.ngrams or not self.vocab:
            return ""
            
        # Set random seed if provided
        if seed is not None:
            self.random = random.Random(seed)
        
        # Initialize with start tokens
        result = []
        w1, w2 = self.start_token, self.start_token
        
        # Track recent n-grams to avoid repetition
        recent_ngrams = set()
        max_recent = 15  # Increased from 10 to better detect longer patterns
        
        # Track sentence state
        sentence_ended = False
        quotes_open = False
        
        for i in range(max_length):
            # Get next word with adjusted temperature
            current_temp = temperature
            
            # Slightly reduce temperature for the first few words to get a good start
            if i < 5:
                current_temp = max(0.1, temperature * 0.7)
                
            w3 = self._get_next_word(w1, w2, current_temp)
            
            # Handle end of text
            if w3 == self.end_token:
                # Only end if we have some content
                if len(result) > 5:  # Require at least 5 words
                    break
                else:
                    # Try a different word
                    w3 = self._get_next_word(w1, w2, current_temp * 1.5)
                    if w3 == self.end_token:
                        w3 = self.random.choice(list(self.vocab))
            
            # Check for repetition
            current_ngram = (w1, w2, w3)
            if current_ngram in recent_ngrams and len(result) > 3:
                # If repeating, increase temperature to break the pattern
                w3 = self._get_next_word(w1, w2, min(1.5, temperature * 1.5))
                
            recent_ngrams.add(current_ngram)
            if len(recent_ngrams) > max_recent:
                recent_ngrams.pop()
            
            # Handle sentence boundaries
            if w3 in ['.', '!', '?']:
                sentence_ended = True
            elif w3 in ['"', "'"]:
                quotes_open = not quotes_open
            
            # Add to result
            result.append(w3)
            
            # Update context
            w1, w2 = w2, w3
            
            # Occasionally end generation at sentence boundaries
            if sentence_ended and len(result) > max_length * 0.7 and self.random.random() > 0.7:
                if not quotes_open:  # Don't end in the middle of a quote
                    break
        
        # Join words with proper spacing
        text = ' '.join(result)
        
        # Post-processing for better readability
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)  # Add space after punctuation if missing
        
        # Handle quotes
        text = re.sub(r'"\s+', '"', text)  # Remove spaces after opening quotes
        text = re.sub(r'\s+"', '"', text)  # Remove spaces before closing quotes
        
        # Capitalization
        if text:
            # Capitalize first letter
            text = text[0].upper() + text[1:]
            
            # Capitalize after sentence boundaries
            for punct in ['. ', '! ', '? ']:
                parts = text.split(punct)
                for i in range(1, len(parts)):
                    if parts[i]:
                        parts[i] = parts[i][0].upper() + parts[i][1:] if parts[i] else ''
                text = punct.join(parts)
            
            # Ensure proper sentence ending
            if text[-1] not in '.!?"\'':
                if text[-1] in ',;:':
                    text = text[:-1] + '.'
                else:
                    text += '.'
            
            # Ensure quotes are balanced
            if text.count('"') % 2 != 0:
                text += '"'
            
            # Remove any double spaces
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text if text.strip() else "I couldn't generate any text. The model might need more training data."
        
    def save(self, filepath: str) -> None:
        """
        Saves the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        import gzip
        
        # Convert defaultdict to regular dict for serialization
        model_data = {
            'ngrams': {k: dict(v) for k, v in self.ngrams.items()},
            'bigrams': dict(self.bigrams),
            'vocab': list(self.vocab),
            'start_token': self.start_token,
            'end_token': self.end_token,
            'unknown_token': self.unknown_token
        }
        
        # Save with gzip compression
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str, random_seed: int = None) -> 'TrigramModel':
        """
        Loads a model from a file.
        
        Args:
            filepath: Path to the saved model
            random_seed: Optional random seed for reproducibility
            
        Returns:
            TrigramModel: Loaded model instance
        """
        import pickle
        import gzip
        from collections import defaultdict
        
        with gzip.open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = cls()
        
        # Set random seed if provided
        if random_seed is not None:
            model.random = random.Random(random_seed)
        
        # Restore model state
        model.ngrams = defaultdict(lambda: defaultdict(int))
        for k, v in model_data['ngrams'].items():
            model.ngrams[k].update(v)
            
        model.bigrams = defaultdict(int, model_data['bigrams'])
        model.vocab = set(model_data['vocab'])
        model.start_token = model_data['start_token']
        model.end_token = model_data['end_token']
        model.unknown_token = model_data['unknown_token']
        
        return model
