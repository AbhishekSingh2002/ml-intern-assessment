import requests
import re
import os

def download_alice():
    """Download and clean Alice in Wonderland text from Project Gutenberg."""
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    output_file = os.path.join(os.path.dirname(__file__), "alice_clean_v2.txt")
    
    print(f"Downloading Alice in Wonderland from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    text = response.text
    
    # Find the actual start and end of the book
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Warning: Could not find start/end markers, using full text")
        content = text
    else:
        content = text[start_idx:end_idx]
    
    # Remove the Project Gutenberg header
    content = content.split('\n\n', 1)[1] if '\n\n' in content else content
    
    # Remove chapter headings
    content = re.sub(r'CHAPTER [IVXLCDM]+\s*\n', '\n\n', content, flags=re.IGNORECASE)
    
    # Remove illustration markers
    content = re.sub(r'\[Illustration:.*?\]', '', content, flags=re.DOTALL)
    
    # Remove multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove special characters but keep basic punctuation
    content = re.sub(r'[^\w\s\.,!?\-\'\"]', ' ', content)
    
    # Normalize whitespace
    content = ' '.join(content.split())
    
    # Basic punctuation cleanup
    content = re.sub(r'\s+([\.,!?])', r'\1', content)
    content = re.sub(r'([\.,!?])(?=[^ ])', r'\1 ', content)
    
    # Save the cleaned content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Cleaned text saved to {output_file}")
    return output_file

if __name__ == "__main__":
    download_alice()
