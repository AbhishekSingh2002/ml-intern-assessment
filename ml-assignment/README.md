# Trigram Language Model

A Python implementation of a trigram language model for text generation. This model learns the statistical relationships between words in a text corpus and can generate new, similar text based on the learned patterns.

## Features

- Train on any text corpus
- Generate new text with customizable length
- Save and load trained models
- Command-line interface for easy usage
- Unit tested with high coverage

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Using pip

```bash
# Install from source
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### As a Python Package

```python
from trigram_lm import TrigramModel

# Create and train a model
model = TrigramModel()
model.fit("This is some example text. The model will learn from this.")

# Generate new text
generated_text = model.generate(length=50)
print(generated_text)

# Save the model
model.save("model.pkl.gz")

# Load the model
loaded_model = TrigramModel.load("model.pkl.gz")
```

### Command Line Interface

```bash
# Train a new model and save it
trigram --train input.txt --save-model model.pkl.gz

# Generate text using a trained model
trigram --load-model model.pkl.gz --length 100 --output output.txt

# Train and generate in one command
trigram --train input.txt --length 100
```

## Project Structure

```
.
├── data/                   # Example training data
│   └── example_corpus.txt
├── src/                    # Source code
│   ├── __init__.py
│   ├── ngram_model.py      # Core TrigramModel implementation
│   └── generate.py         # Command-line interface
├── tests/                  # Unit tests
│   └── test_ngram.py
├── setup.py                # Package configuration
└── README.md               # This file
```

## Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd trigram-lm

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

This project uses:
- Black for code formatting
- Flake8 for linting
- isort for import sorting
- mypy for type checking

Run the following to format and check the code:

```bash
black src/
isort src/
flake8 src/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Design Choices

### Model Architecture
- **N-gram Size**: Chose trigrams (3-grams) as they provide a good balance between context and data sparsity
- **Smoothing**: Implemented basic smoothing by falling back to vocabulary sampling for unseen contexts
- **Tokenization**: Simple word-level tokenization with basic punctuation handling

### Implementation Details
- **Data Structures**: Used nested dictionaries for efficient n-gram counting and lookup
- **Serialization**: Implemented model saving/loading using Python's pickle with gzip compression
- **Error Handling**: Added comprehensive error handling and input validation

### Performance Considerations
- **Memory Usage**: Optimized for memory efficiency with generator-based text generation
- **Speed**: Implemented efficient dictionary lookups for n-gram probabilities

See [evaluation.md](evaluation.md) for more detailed discussion of design decisions and their trade-offs.
