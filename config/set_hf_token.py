import os
import sys
import re

def set_token_in_file(file_path, token):
    """Replace placeholder with actual token in a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the placeholder with the actual token
    updated_content = content.replace('"YOUR_HUGGINGFACE_TOKEN_HERE"', f'"{token}"')
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated token in {file_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python set_hf_token.py YOUR_HUGGINGFACE_TOKEN")
        sys.exit(1)
    
    token = sys.argv[1]
    
    # Validate token format (basic check)
    if not re.match(r'^hf_[a-zA-Z0-9]+$', token):
        print("Warning: Token doesn't match expected format (hf_XXXXXXXX). Continue anyway? (y/n)")
        response = input().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(1)
    
    # Update token in files
    files_to_update = [
        "load_models.py",
        "multi_model_system.py"
    ]
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            set_token_in_file(file_path, token)
        else:
            print(f"Warning: File {file_path} not found")
    
    print("\nToken has been set in all files.")
    print("You can now run the pipeline with access to the LLaMA model.")

if __name__ == "__main__":
    main()
