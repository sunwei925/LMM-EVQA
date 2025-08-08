# LMM-EVQA
![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/LMM-EVQA) [![](https://img.shields.io/github/stars/sunwei925/LMM-EVQA)](https://github.com/sunwei925/LMM-EVQA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/LMM-EVQA)
[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2508.02516)


🏆 🥇 **Winner solution for [ICCV VQualA 2025 EVQA-SnapUGC: Engagement prediction for short videos Challenge](https://codalab.lisn.upsaclay.fr/competitions/23005) at the [VQualA 2025](https://vquala.github.io/) workshop @ ICCV 2025** 

Official Code for **Engagement Prediction of Short Videos with Large Multimodal Models**

## Introduction
> The rapid proliferation of user-generated content (UGC) on short-form video platforms has made video engagement prediction increasingly important for optimizing recommendation systems and guiding content creation. However, this task remains challenging due to the complex interplay of factors such as semantic content, visual quality, audio characteristics, and user background. Prior studies have leveraged various types of features from different modalities, such as visual quality, semantic content, background sound, etc., but often struggle to effectively model their cross-feature and cross-modality interactions. In this work, we empirically investigate the potential of large multimodal models (LMMs) for video engagement prediction. We adopt two representative LMMs: **VideoLLaMA2**, which integrates audio, visual, and language modalities, and **Qwen2.5-VL**, which models only visual and language modalities. Specifically, VideoLLaMA2 jointly processes key video frames, text-based metadata, and background sound, while Qwen2.5-VL utilizes only key video frames and text-based metadata. Trained on the SnapUGC dataset, both models demonstrate competitive performance against state-of-the-art baselines, showcasing the effectiveness of LMMs in engagement prediction. Notably, VideoLLaMA2 consistently outperforms Qwen2.5-VL, highlighting the importance of audio features in engagement prediction. By ensembling two types of models, our method achieves **first place in the ICCV VQualA 2025 EVQA-SnapUGC Challenge on short-form video engagement prediction**.

## Model
![Model Figure](./figures/framework.PNG)

#### Performance on ICCV VQualA 2025 EVQA Challenge
| **Team name**                    | **Final Score** | **SROCC** | **PLCC**  |
|-----------------------------|-------------|--------|--------|
| **ECNU-SJTU VQA (ours)**        | **0.710**       | **0.707**  | **0.714**  |
| IMCL-DAMO                   | 0.698       | 0.696  | 0.702  |
| HKUST-Cardiff-MI-BAAI       | 0.680       | 0.677  | 0.684  |
| MCCE                        | 0.667       | 0.660  | 0.668  |
| EasyVQA                     | 0.667       | 0.664  | 0.671  |
| Rochester                   | 0.449       | 0.405  | 0.515  |
| brucelyu                    | 0.441       | 0.439  | 0.444  |


**Table:** Result of VQualA 2025 EVQA-SnapUGC Challenge. The final score is computed as 0.6xSRCC+0.4xPLCC.


- for more results on the ICCV VQualA 2025 EVQA Challenge, please refer to the challenge report.

## Dataset

### Download Links

#### CSV Files
- **Training CSV**: [Google Drive](https://drive.google.com/file/d/1Mv5Esq5gGuxRTayabRUb5NmHwEN3JdbD/view) | [Baidu Yun](https://pan.baidu.com/s/1iY7qUfqivYzUqUID501dKQDpI3NQil7y/view)
- **Validation CSV**: [Google Drive](https://drive.google.com/file/d/1iY7qUfqivYzUqUID501dKQDpI3NQil7y/view) | [Baidu Yun](https://pan.baidu.com/s/1iY7qUfqivYzUqUID501dKQDpI3NQil7y/view)

#### Video Files
- **Training Videos**: [Google Drive](https://drive.google.com/drive/folders/134gJflcaQ7Dhg5EUKfLdeXW61fj1fiNo?usp=share_link) | [Baidu Yun](https://pan.baidu.com/s/18nk2BzrykyHusfTDX5w7xg?pwd=edts) (extraction code: `edts`)
- **Validation Videos**: [Google Drive](https://drive.google.com/file/d/15pIdfXoOGnTye-99K8Wror7sMNU0uzh-/view?usp=share_link) | [Baidu Yun](https://pan.baidu.com/s/1UtTXwgb13B7lxDFjtLPx5g) (extraction code: `28tq`)

### Dataset Structure
```
SnapUGC/
├── train_data.csv          # Training annotations
├── val_data.csv            # Validation annotations  
├── train_videos/           # Training video files
└── val_videos/            # Validation video files
```

## VideoLLaMA2
### Training
- Navigate to the project directory
```
cd VideoLLaMA2-audio_visual
```

- Set up the computational environment
```
conda create -n videollama2 python=3.9
conda activate videollama2
pip install requirements.txt
```

- Download pre-trained model weights for VideoLLaMA2-audio_visual
```
python download_model_weight.py
--local_dir specifies the target directory for model weights
```

- Preprocess and prepare training dataset
```
python prepare_dataset.py
VIDEO_DIR = " " # Path to video directory
AUDIO_DIR = " " # Audio directory parameter (deprecated) - VideoLLaMA2 automatically extracts audio features from video files
CSV_PATH = "train_data.csv" # Input CSV file containing video metadata
OUTPUT_TRAIN = "train.json" # Output JSON file for training data
```

- Execute model training
```
sh train.sh
--model_path # Path to pre-trained model weights
--data_path # Path to training data JSON file
--num_frames # Number of frames to extract from each video (first N frames)
--output_dir # Directory for saving trained model weights
--num_train_epochs # Total number of training epochs
--per_device_train_batch_size # Batch size per GPU device
```

### Testing
- Pre-trained model weights are available for download via [Baidu Yun](https://pan.baidu.com/s/17AGQyy-6QKY5V7C0BjVZeA) (extraction code: 3aqc)

- Evaluate model performance on the validation set of SnapUGC dataset
```
# Generate validation dataset using the same preprocessing pipeline
python prepare_dataset.py
```

```
sh run_validation.sh
--model-path # Path to the trained model weights
--modal-type # Specify "av" mode for audio-visual evaluation
--json_file # Path to the validation JSON file
```

### Single Video Testing
- Test engagement prediction for individual videos with custom title and description
```
python videollama2/test_single_video.py \
    --model-path /path/to/trained/model \
    --modal-type av \
    --video-path /path/to/video.mp4 \
    --title "Video Title" \
    --description "Video Description"
```

**Parameters:**
- `--model-path`: Path to the trained model weights (required)
- `--modal-type`: Modal type - "a" (audio only), "v" (video only), "av" (audio-visual)
- `--video-path`: Path to the video file (required)
- `--title`: Title of the video (optional, default: None)
- `--description`: Description of the video content (optional, default: None)

**Example Usage:**
```bash
# Test with video path only
python videollama2/test_single_video.py \
    --model-path /path/to/trained/model \
    --modal-type av \
    --video-path /path/to/your/video.mp4

# Test with title and description
python videollama2/test_single_video.py \
    --model-path /path/to/trained/model \
    --modal-type av \
    --video-path /path/to/your/video.mp4 \
    --title "Your Video Title" \
    --description "Your Video Description"
```


## Citation
**If you find this code is useful for  your research, please cite**:

```latex
@inproceedings{sun2025engagement,
title={Engagement Prediction of Short Videos with Large Multimodal Models},
  author={Sun, Wei and Cao, Linhan and Cao, Yuqin and Zhang, Weixia and Wen, Wen and Zhang, Kaiwei, and Chen, Zijian and Lu, Fangfang and Min, Xiongkuo and Zhai, Guangtao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision (ICCV) Workshops},
  pages = {1-10},
  year={2025}
}
```