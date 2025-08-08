import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from vision_process import process_vision_info
import json
import torch
import argparse
import pandas as pd

def create_conversation(title, description):
    """Create conversation format with title and description."""
    title = "None" if pd.isna(title) else title
    description = "None" if pd.isna(description) else description
    
    return [
        {
            "from": "human",
            "value": f"<video>\nHow would you judge the engagement continuation rate of the given content, where engagement continuation rate represents the probability of watch time exceeding 5 seconds. The title of the video is {title}, the description of the video is {description}"
        },
        {
            "from": "gpt",
            "value": "The engagement continuation rate of the video."
        }
    ]

def inference_single_video(args):
    # Load model and processor
    model_path = os.path.expanduser(args.model_path)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Create conversation with video, title, and description
    video_path = args.video_path
    title = args.title
    description = args.description
    
    # Create conversation
    conversation = create_conversation(title, description)
    
    # Create sample data structure
    sample = {
        "video": video_path,
        "conversations": conversation
    }
    
    print(f"\n=== Qwen2.5-VL Engagement Prediction Results ===")
    print(f"Video Path: {video_path}")
    if title and title != "None":
        print(f"Title: {title}")
    if description and description != "None":
        print(f"Description: {description}")
    
    # Extract question from conversation
    qs = sample["conversations"][0]["value"][7:]  # Remove "<video>\n" prefix
    
    # Create messages for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "fps": 1.0,
                },
                {"type": "text", "text": qs},
            ],
        }
    ]
    
    print(f"Processing video: {video_path}")
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    prediction = output_text[0]
    print(f"Predicted Engagement Continuation Rate: {prediction}")
    print(f"Score Range: 0-100 (probability of watch time exceeding 5 seconds)")
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test engagement prediction for a single video using Qwen2.5-VL")
    parser.add_argument('--model-path', required=True, 
                       help='Path to the trained Qwen2.5-VL model weights')
    parser.add_argument('--video-path', required=True, 
                       help='Path to the video file')
    parser.add_argument('--title', default=None, 
                       help='Title of the video (optional)')
    parser.add_argument('--description', default=None, 
                       help='Description of the video content (optional)')
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} does not exist!")
        exit(1)
    
    # Run inference
    engagement_score = inference_single_video(args) 