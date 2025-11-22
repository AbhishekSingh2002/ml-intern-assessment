import os
import argparse
import sys
from pathlib import Path
from .ngram_model import TrigramModel

def train_model(input_file, model_save_path=None):
    """
    Train a new model on the given input file and optionally save it.
    
    Args:
        input_file (str): Path to the training text file
        model_save_path (str, optional): Path to save the trained model
        
    Returns:
        TrigramModel: The trained model
    """
    model = TrigramModel()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Training model on {input_file}...")
        model.fit(text)
        print("Training completed successfully!")
        
        if model_save_path:
            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
            
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        return None
        
    return model

def generate_text(model, length=100, output_file=None, temperature=0.8, seed=None):
    """
    Generate text using the provided model.
    
    Args:
        model (TrigramModel): The trained model
        length (int): Maximum number of words to generate
        output_file (str, optional): File to save the generated text
        temperature (float): Controls randomness (0.1-1.0, higher is more random)
        seed (int, optional): Random seed for reproducibility
    """
    try:
        print(f"\nGenerating text (temperature={temperature}, seed={seed if seed is not None else 'random'})...")
        generated_text = model.generate(max_length=length, temperature=temperature, seed=seed)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            print(f"Generated text saved to {output_file}")
        else:
            print("\n" + "="*50)
            print(generated_text)
            print("="*50 + "\n")
            
    except Exception as e:
        print(f"Error during text generation: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Train and generate text using a Trigram Language Model')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--train', type=str, help='Train the model on the given input file')
    train_group.add_argument('--save-model', type=str, help='Path to save the trained model')
    
    # Generation arguments
    gen_group = parser.add_argument_group('Generation')
    gen_group.add_argument('--load-model', type=str, help='Load a pre-trained model from file')
    gen_group.add_argument('--length', type=int, default=100, help='Maximum number of words to generate (default: 100)')
    gen_group.add_argument('--output', type=str, help='Output file for generated text (default: print to console)')
    gen_group.add_argument('--temperature', type=float, default=0.8, help='Controls randomness (0.1-1.0, higher is more random, default: 0.8)')
    gen_group.add_argument('--seed', type=int, help='Random seed for reproducibility (default: random)')
    
    args = parser.parse_args()
    
    # Validate temperature
    if args.temperature <= 0 or args.temperature > 2.0:
        parser.error("Temperature must be between 0 and 2.0")
    
    if not any([args.train, args.load_model]):
        parser.print_help()
        print("\nError: Either --train or --load-model must be specified")
        sys.exit(1)
    
    model = None
    
    # Load or train model
    if args.load_model:
        try:
            model = TrigramModel()
            model.load(args.load_model)
            print(f"Model loaded from {args.load_model}")
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
    
    if args.train:
        model = train_model(args.train, args.save_model)
        if model is None:
            sys.exit(1)
    
    # Generate text if we have a model
    if model:
        generate_text(model, args.length, args.output, args.temperature, args.seed)

if __name__ == "__main__":
    main()
