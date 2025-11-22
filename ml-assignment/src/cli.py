import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train and generate text using a trigram language model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--train', required=True, help='Path to training text file')
    train_parser.add_argument('--save-model', required=True, help='Path to save the trained model')
    train_parser.add_argument('--min-count', type=int, default=1, help='Minimum word count to include in vocabulary')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text using a trained model')
    gen_parser.add_argument('--load-model', required=True, help='Path to the trained model')
    gen_parser.add_argument('--length', type=int, default=100, help='Maximum number of words to generate')
    gen_parser.add_argument('--temperature', type=float, default=0.8, 
                          help='Temperature for generation (higher = more random)')
    gen_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    gen_parser.add_argument('--output', type=str, help='Output file to save generated text')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a trained model')
    inspect_parser.add_argument('model_path', help='Path to the trained model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from .ngram_model import TrigramModel
        
        print(f"Training model on {args.train}...")
        with open(args.train, 'r', encoding='utf-8') as f:
            text = f.read()
            
        model = TrigramModel()
        model.fit(text, min_word_count=args.min_count)
        
        # Create directory if it doesn't exist
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(str(model_path))
        print(f"Model saved to {model_path}")
        
    elif args.command == 'generate':
        from .ngram_model import TrigramModel
        
        print(f"Loading model from {args.load_model}...")
        model = TrigramModel.load(args.load_model, random_seed=args.seed)
        
        print("\nGenerating text...")
        text = model.generate(max_length=args.length, temperature=args.temperature, seed=args.seed)
        
        if args.output:
            # Create directory if it doesn't exist
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Generated text saved to {output_path}")
        else:
            print("\n" + "=" * 50)
            print(text)
            print("=" * 50 + "\n")
        
    elif args.command == 'inspect':
        from .ngram_model import TrigramModel
        
        print(f"Inspecting model at {args.model_path}")
        model = TrigramModel.load(args.model_path)
        
        print(f"\nVocabulary size: {len(model.vocab)}")
        print(f"Number of trigram contexts: {len(model.ngrams)}")
        
        # Count total n-grams
        total_ngrams = sum(len(context) for context in model.ngrams.values())
        print(f"Total n-grams: {total_ngrams}")
        
        # Check special tokens
        print("\nSpecial tokens:")
        for token in [model.start_token, model.end_token, model.unknown_token]:
            print(f"  {token}: {token in model.vocab}")
            
        # Show most common contexts
        print("\nMost common contexts:")
        sorted_contexts = sorted(
            model.ngrams.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )
        for (w1, w2), words in sorted_contexts[:5]:
            total = sum(words.values())
            top_words = sorted(words.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{w} ({c/total:.1%})" for w, c in top_words)
            print(f"  '{w1} {w2}' -> {top_str} (total: {total})")
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
