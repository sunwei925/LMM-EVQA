import json
import os

def extract_samples(input_file, output_file, num_samples=10000):
    """
    Extract the first num_samples from input_file and save to output_file
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        num_samples (int): Number of samples to extract
    """
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract first num_samples
    extracted_data = data[:num_samples]
    print(f"Extracted {len(extracted_data)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to output file
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "/root/workspace/cvuaggk7v38s73dgjft0/code/VideoLLaMA2-audio_visual/train.json"
    output_file = "/root/workspace/cvuaggk7v38s73dgjft0/code/VideoLLaMA2-audio_visual/train_10k.json"
    extract_samples(input_file, output_file) 