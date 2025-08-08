import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from decord import VideoReader, cpu

# Define paths
VIDEO_DIR = "/root/workspace/cvuaggk7v38s73dgjft0/EVQA_SnapUGC/train_dataset/train_videos"
CSV_PATH = "train_data.csv"
OUTPUT_TRAIN = "train_qwenvl.json"

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



def read_txt_file(file_path):
    """
    读取文本文件的内容
    
    Args:
        file_path (str): 文本文件的路径
        
    Returns:
        str: 文件内容
        
    Raises:
        FileNotFoundError: 如果文件不存在
        Exception: 其他可能的错误
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def create_conversation(title, description, ECR):
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
            "value": f"{ECR*100}."
        }
    ]

def create_dataset_entry(row):
    """Create a single dataset entry from a row."""
    video_path = os.path.join(VIDEO_DIR, f"{row['Id']}.mp4")
    
    # 检查视频是否可用
    if not check_video(video_path):
        return None
        
    return {
        "video": video_path,
        "conversations": create_conversation(row["Title"], row["Description"], row["ECR"])
    }

def main():
    # Read CSV file
    train_df = pd.read_csv(CSV_PATH)
    
    
    # Process training set
    train_entries = []
    train_total = len(train_df)
    train_valid = 0
    
    print("Processing training samples...")
    for _, row in train_df.iterrows():
        entry = create_dataset_entry(row)
        if entry is not None:
            train_entries.append(entry)
            train_valid += 1
            
        if train_valid % 100 == 0:
            print(f"Processed {train_valid} valid training samples out of {train_total} total samples")
    
    print(f"\nTraining set:")
    print(f"Total samples: {train_total}")
    print(f"Valid samples: {train_valid}")
    print(f"Invalid/skipped samples: {train_total - train_valid}")
    
    
    # Save to JSON files
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=4)
    
    print(f"\nCreated training dataset with {len(train_entries)} samples")

if __name__ == "__main__":
    main()