#!/bin/bash

# Example script for testing engagement prediction on a single video
# Usage: ./run_single_video_test.sh

# Example usage with video path only (no title/description)
python videollama2/test_single_video.py \
    --model-path /root/workspace/cvuaggk7v38s73dgjft0/videollama2_EVQA_weights_mse \
    --modal-type av \
    --video-path /path/to/your/video.mp4

# Example usage with title and description
# python videollama2/test_single_video.py \
#     --model-path /root/workspace/cvuaggk7v38s73dgjft0/videollama2_EVQA_weights_mse \
#     --modal-type av \
#     --video-path /path/to/your/video.mp4 \
#     --title "Amazing Cooking Tutorial" \
#     --description "Learn how to make delicious pasta from scratch with step-by-step instructions"

echo "Single video engagement test completed!" 