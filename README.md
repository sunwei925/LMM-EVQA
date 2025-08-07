# LMM-EVQA
![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/LMM-EVQA) [![](https://img.shields.io/github/stars/sunwei925/LMM-EVQA)](https://github.com/sunwei925/LMM-EVQA)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/LMM-EVQA)
[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2508.02516)


🏆 🥇 **Winner solution for [ICCV VQualA 2025 EVQA-SnapUGC: Engagement prediction for short videos Challenge](https://codalab.lisn.upsaclay.fr/competitions/23005) at the [VQualA 2025](https://vquala.github.io/) workshop @ ICCV 2025** 

Official Code for **Engagement Prediction of Short Videos with Large Multimodal Models**

## Introduction
> The rapid proliferation of user-generated content (UGC) on short-form video platforms has made video engagement prediction increasingly important for optimizing recommendation systems and guiding content creation. However, this task remains challenging due to the complex interplay of factors such as semantic content, visual quality, audio characteristics, and user background. Prior studies have leveraged various types of features from different modalities, such as visual quality, semantic content, background sound, etc., but often struggle to effectively model their cross-feature and cross-modality interactions. In this work, we empirically investigate the potential of large multimodal models (LMMs) for video engagement prediction. We adopt two representative LMMs: **VideoLLaMA2**, which integrates audio, visual, and language modalities, and **Qwen2.5-VL**, which models only visual and language modalities. Specifically, VideoLLaMA2 jointly processes key video frames, text-based metadata, and background sound, while Qwen2.5-VL utilizes only key video frames and text-based metadata. Trained on the SnapUGC dataset, both models demonstrate competitive performance against state-of-the-art baselines, showcasing the effectiveness of LMMs in engagement prediction. Notably, VideoLLaMA2 consistently outperforms Qwen2.5-VL, highlighting the importance of audio features in engagement prediction. By ensembling two types of models, our method achieves **first place in the ICCV VQualA 2025 EVQA-SnapUGC Challenge on short-form video engagement prediction**.

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

## The training and test code will be released soon!

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