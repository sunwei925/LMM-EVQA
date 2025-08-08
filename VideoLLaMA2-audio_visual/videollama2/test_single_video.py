import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./')
from videollama2 import model_init
from videollama2.mm_utils import tokenizer_multimodal_token
import argparse
import json
import torch
import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict    
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import scipy.io
from transformers.trainer_pt_utils import (nested_detach)
from tqdm import tqdm
import pandas as pd
import scipy.stats

def preprocess_plain(source, tokenizer, modal_token: str = None):
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    assert len(source) == 2
    assert modal_token in source[0]['value']
    message = [
        {'role': 'user', 'content': source[0]['value']},
        {'role': 'assistant', 'content': source[1]['value']}
    ]
    conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
    return dict(input_ids=input_ids) #, labels=targets

def collator(inputs, device):
    batch = dict(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs['attention_mask'],
    )
    batch['images'] = []
    batch['images'].append((inputs['video'], 'video'))
    return batch

def inference_single_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)
    model.get_tokenizer(tokenizer)
    
    # Set modal type
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError("Modal type must be 'a', 'v', or 'av'")

    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    
    # Create conversation with video, title, and description
    video_path = args.video_path
    title = args.title
    description = args.description
    
    # Construct the prompt with video, title, and description
    title = "None" if title is None else title
    description = "None" if description is None else description
    
    prompt = f"How would you judge the engagement continuation rate of the given content, where engagement continuation rate represents the probability of watch time exceeding 5 seconds. The title of the video is {title}, and the description of the video is {description}"
    
    conversation = [
        {
            "from": "human",
            "value": f"<video>\n{prompt}"
        },
        {
            "from": "gpt",
            "value": "The engagement continuation rate of the video."
        }
    ]

    modal_token = "<video>"
    data_dict = preprocess_plain(conversation, tokenizer, modal_token=modal_token)
    data_dict["input_ids"] = data_dict["input_ids"][0].unsqueeze(0)
    data_dict['attention_mask'] = data_dict["input_ids"][0].ne(tokenizer.pad_token_id).unsqueeze(0)

    # Process video/audio
    if args.modal_type == "a":
        audio_video_tensor = preprocess(video_path)
    else:
        audio_video_tensor = preprocess(video_path, va=True if args.modal_type == "av" else False)
    
    audio_video_tensor['video'] = audio_video_tensor['video'].half().to(device)
    audio_video_tensor['audio'] = audio_video_tensor['audio'].half().to(device)
    data_dict['video'] = audio_video_tensor

    inputs = collator(data_dict, device)
    
    with torch.no_grad():
        outputs = model(**inputs, return_dict=False)
    
    # Get prediction
    prediction = outputs[0].cpu().numpy()
    
    print(f"\n=== Engagement Continuation Rate Prediction Results ===")
    print(f"Video Path: {video_path}")
    if title and title != "None":
        print(f"Title: {title}")
    if description and description != "None":
        print(f"Description: {description}")
    print(f"Modal Type: {args.modal_type}")
    print(f"Predicted Engagement Continuation Rate: {prediction[0]:.4f}")
    print(f"Score Range: 0-100 (probability of watch time exceeding 5 seconds)")
    
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test engagement prediction for a single video")
    parser.add_argument('--model-path', default='/root/workspace/cvuaggk7v38s73dgjft0/videollama2_EVQA_weights_mse', 
                       help='Path to the trained model weights')
    parser.add_argument('--modal-type', default='av', choices=["a", "v", "av"], 
                       help='Modal type: a=audio only, v=video only, av=audio-visual')
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
        sys.exit(1)
    
    # Run inference
    engagement_score = inference_single_video(args) 