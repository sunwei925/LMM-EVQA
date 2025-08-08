import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from decord import VideoReader, cpu

# Define paths
VIDEO_DIR = "/root/workspace/cvuaggk7v38s73dgjft0/EVQA_SnapUGC/train_dataset/train_videos"
AUDIO_DIR = "/root/workspace/cvuaggk7v38s73dgjft0/EVQA_SnapUGC/train_dataset/train_audios"
CSV_PATH = "train_data.csv"
OUTPUT_TRAIN = "train.json"

def check_video(video_path):
    """检查视频文件是否存在且可以正常读取
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        bool: 视频是否可用
    """
    try:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
            
        # 尝试读取视频
        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) == 0:
            print(f"Empty video file: {video_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Error reading video {video_path}: {str(e)}")
        return False

def create_conversation(title, description):
    """Create conversation format with title and description."""
    title = "None" if pd.isna(title) else title
    description = "None" if pd.isna(description) else description
    
    return [
        {
            "from": "human",
            "value": f"<video>\nHow would you judge the engagement continuation rate of the given content, where engagement continuation rate represents the probability of watch time exceeding 5 seconds. The title of the video is {title}, and the description of the video is {description}"
        },
        {
            "from": "gpt",
            "value": "The engagement continuation rate of the video."
        }
    ]

def create_dataset_entry(row):
    """Create a single dataset entry from a row."""
    video_path = os.path.join(VIDEO_DIR, f"{row['Id']}.mp4")
    
    # 检查视频是否可用
    if not check_video(video_path):
        return None
        
    return {
        "id": row["Id"],
        "ECR": row["ECR"]*100,
        "video": video_path,
        "audio": os.path.join(AUDIO_DIR, f"{row['Id']}.wav"),
        "conversations": create_conversation(row["Title"], row["Description"])
    }

def main():
    # Read CSV file
    df = pd.read_csv(CSV_PATH)
    
    # Create dataset entries with video validation
    all_entries = []
    total_samples = len(df)
    valid_samples = 0
    
    print("Processing samples and validating videos...")
    for _, row in df.iterrows():
        entry = create_dataset_entry(row)
        if entry is not None:
            all_entries.append(entry)
            valid_samples += 1
            
        if valid_samples % 100 == 0:
            print(f"Processed {valid_samples} valid samples out of {total_samples} total samples")
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Valid samples: {valid_samples}")
    print(f"Invalid/skipped samples: {total_samples - valid_samples}")
    
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=4)
    
    print(f"\nCreated validation dataset with {len(all_entries)} samples")

if __name__ == "__main__":
    main() 