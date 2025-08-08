#!/bin/bash

# Example script for testing engagement prediction on a single video using Qwen2.5-VL
# Usage: ./run_single_video_test_qwenvl.sh

# Example usage with video path only (no title/description)
python test_single_video_qwenvl.py \
    --model-path /path/to/trained/qwenvl/model \
    --video-path /path/to/your/video.mp4

# Example usage with title and description
# python test_single_video_qwenvl.py \
#     --model-path /path/to/trained/qwenvl/model \
#     --video-path /path/to/your/video.mp4 \
#     --title "Your Video Title" \
#     --description "Your Video Description"

echo "Qwen2.5-VL single video engagement test completed!" 