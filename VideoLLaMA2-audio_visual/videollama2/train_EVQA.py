# Adopted from https://github.com/haotian-liu/LLaVA and other sources
# 主要用于训练支持三分制评分的视频质量评估模型

# 基础库导入
import re
import os
import copy
import json
import numpy as np
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# 科学计算相关
import scipy.stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# torch相关
import torch
from torch.utils.data import Dataset
import transformers
import deepspeed
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

# 项目内部导入
import sys
sys.path.append('./')
from videollama2.model import *
from videollama2.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from videollama2.mm_utils import tokenizer_multimodal_token, process_video, process_image, process_audio_file
from videollama2.videollama2_trainer_EVQA import (VideoLLaMA2Trainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

# 设置tokenizer并行处理参数，避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None

def rank0_print(*args):
    """仅在主进程(rank 0)上打印日志
    
    Args:
        *args: 要打印的参数
    """
    if local_rank == 0:
        print(*args)

def set_seed(seed=42):
    """设置随机种子以保证结果可复现
    
    Args:
        seed: 随机种子值，默认为42
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 用于多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ModelArguments:
    """模型相关的参数配置类
    
    包含LLM模型、连接器、视觉塔和音频塔的配置参数
    """
    # LLM参数
    model_type: Optional[str] = field(default="videollama2", 
                                    metadata={"help": "模型类型，可选值: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", 
                                 metadata={"help": "会话模板的版本"})
    freeze_backbone: bool = field(default=False, 
                                metadata={"help": "是否冻结LLM骨干网络"})
    # add
    tune_adapter_llm: bool = field(default=False)
    
    # 连接器参数
    mm_projector_type: Optional[str] = field(default='linear')
    mm_projector_a_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_adapter_a: bool = field(default=False) 
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter_a: Optional[str] = field(default=None)
    
    # 视觉塔参数
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    
    # 音频塔参数
    audio_tower: Optional[str] = field(default=None)
    tune_audio_tower: bool = field(default=False)

@dataclass
class DataArguments:
    """数据相关的参数配置类
    
    包含数据路径、加载和预处理相关的参数
    """
    # 路径参数
    data_path: str = field(default=None, 
                          metadata={"help": "训练数据的路径"})
    data_path_a: Optional[str] = field(default=None, 
                                     metadata={"help": "音频数据的路径"})
    data_folder: Optional[str] = field(default=None)
    
    # 加载参数
    is_multimodal: bool = False  # 是否是多模态数据
    va: bool = field(default=False)  # 是否包含视觉和音频
    lazy_preprocess: bool = False  # 是否使用懒加载预处理
    num_frames: Optional[int] = field(default=None)  # 视频帧数
    
    # 预处理参数
    image_aspect_ratio: str = 'square'  # 图像长宽比处理方式

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """训练相关的参数配置类
    
    继承自transformers的TrainingArguments，添加了一些自定义参数
    """
    optim: str = field(default="adamw_torch")  # 优化器类型
    mm_projector_lr: Optional[float] = None  # 多模态投影层的学习率
    freeze_mm_mlp_adapter: bool = field(default=False)  # 是否冻结多模态MLP适配器
    remove_unused_columns: bool = field(default=False)
    loss_type: str = field(default="mse")
    
    # 训练数据参数
    group_by_modality_length: bool = field(default=False)  # 是否按模态长度分组
    model_max_length: int = field(default=512,
        metadata={"help": "最大序列长度，超出部分会被截断"})
    
    # LoRA或量化参数
    double_quant: bool = field(default=True,
        metadata={"help": "是否使用双重量化压缩量化统计信息"})
    quant_type: str = field(default="nf4",
        metadata={"help": "量化类型，可选'fp4'或'nf4'"})
    bits: int = field(default=16,
        metadata={"help": "量化位数"})
    
    # LoRA参数
    lora_enable: bool = False  # 是否启用LoRA
    lora_r: int = 64  # LoRA秩
    lora_alpha: int = 16  # LoRA alpha参数
    lora_dropout: float = 0.05  # LoRA dropout率
    lora_weight_path: str = ""  # LoRA权重路径
    lora_bias: str = "none"  # LoRA偏置类型
    
    # 评估参数
    label_names: Optional[List[str]] = field(default_factory=lambda: ['labels', 'mos'])
    remove_unused_columns = False
    include_inputs_for_metrics = True
    metric_for_best_model: str="SRCC"  # 用于选择最佳模型的指标
    greater_is_better: bool = True  # 指标是否越大越好
    save_safetensors: bool = False
    load_best_model_at_end: bool = False  # 默认不加载最佳模型

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    """预处理单个示例数据
    
    Args:
        sources: 源数据序列
        tokenizer: tokenizer对象
        modal_token: 模态标记(如<video>,<image>等)
        
    Returns:
        包含input_ids和labels的字典
    """
    # 定义评分标记
    toks = [" excellent", " good", " fair", " poor", " bad"]
    ids_ = [tokenizer(id_).input_ids[0] for id_ in toks]

    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    
    for source in sources:
        # 1. 应用对话模板获取输入对话
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': source[0]['value']},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        
        # 2. tokenize对话
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        
        # 3. 生成训练目标
        targets.append(copy.deepcopy(input_ids[-1]))
        instruction = tokenizer.apply_chat_template(message[:1], tokenize=False, add_generation_prompt=True)
        instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
        targets[-1][:instruction_len] = IGNORE_INDEX

        # # 替换特殊token
        # for i in range(instruction_len,input_ids[0].shape[0]):
        #     if input_ids[0][i] in ids_:
        #         input_ids[0][i] = 198
                
    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    """预处理对话数据
    
    Args:
        sources: 源数据序列
        tokenizer: tokenizer对象
        modal_token: 模态标记(如<video>,<image>等)
        
    Returns:
        包含input_ids和labels的字典
    """
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]
        message = [{'role': roles[sentence['from']], 'content': sentence['value']} for sentence in source]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))
        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    {'role': roles[source[idx-1]['from']], 'content': source[idx-1]['value']}, 
                    {'role': roles[sentence['from']], 'content': sentence['value']}
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
                conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

                targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message
    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    modal_token: str = None,
) -> Dict:
    """预处理多模态数据
    
    Args:
        sources: 源数据序列 
        data_args: 数据参数配置
        modal_token: 模态标记
        
    Returns:
        处理后的数据
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    assert modal_token in MODAL_INDEX_MAP, f"不支持的模态标记 {modal_token}"

    # 处理每个样本中的模态标记
    for source in sources:
        for sentence in source:
            if modal_token in sentence['value']:
                # 调整模态标记的位置到句首
                sentence['value'] = sentence['value'].replace(modal_token, '').strip()
                sentence['value'] = modal_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = modal_token
            sentence["value"] = sentence["value"].replace(modal_token, replace_token)

    return sources

import json
class LazySupervisedDataset(Dataset):
    """用于监督训练的数据集类
    
    支持延迟加载和处理数据，提高内存使用效率
    """
    def __init__(self, data_path: str, data_path_a: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.mix_sampler_tag = False  # 是否使用混合采样器
        
        # 加载数据
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.data_args = data_args
        
    def __len__(self):
        return len(self.list_data_dict)
    
    @property
    def lengths(self):
        """获取每个样本的长度"""
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list
    
    @property 
    def modality_lengths(self):
        """获取每个样本在不同模态下的长度"""
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """获取单个训练样本
        
        Args:
            i: 样本索引
            
        Returns:
            包含processed_images/video和对应标签的字典
        """
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # if sources[0]['conversations'][0]['value'].endswith(" "):
        #     question = sources[0]['conversations'][0]['value'][:-1]
        # else:
        #     question = sources[0]['conversations'][0]['value']
        # sources[0]['video'] = sources[0]['video'].replace('/mnt/sda/cyq/Database/AIGC', '/root/autodl-tmp/Database/AIGC')
        # sources[0]['conversations'][0]['value'] = question[:-1].replace('<audio>\n','') #+ " in one word" + question[-1:]
        # sources[0]['conversations'][1]['value'] = sources[0]['conversations'][1]['value']#.split()[-1].capitalize()
        if self.data_args.data_path is not None:
            image_processor = self.data_args.image_processor
            video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.data_folder
            image_file = os.path.join(image_folder, image_file)

            try:
                image = process_image(image_file, image_processor, aspect_ratio=self.data_args.image_aspect_ratio)
            except:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            # place <image> tag to question head.
            # modal_token = "<image>"
            # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            # video_folder = self.data_args.data_folder
            # if video_folder:
            #     video_file = os.path.join(video_folder, video_file)

            try:
                video = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames, va = self.data_args.va if not self.mix_sampler_tag else (i < len(self.av_data)), sample_mode='first_n_seconds')
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            # place <video> tag to question head.
            modal_token = "<video>"
            sources = copy.deepcopy([e["conversations"] for e in sources]) #preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)

        # elif 'audio' in sources[0]:
        #     audio_file = self.list_data_dict[i]['audio']
        #     #audio_folder = self.data_args.base_folder
        #     #print(audio_file)
        #     try:
        #         audio = process_audio_file(audio_file)
        #     except Exception as e:
        #         print(e)
        #         backup_idx = random.randint(0, len(self.list_data_dict)-1)
        #         print(f"Encounted error when reading audio {audio_file}, use {backup_idx}-th example instead!!!")
        #         return self.__getitem__(backup_idx)
            # modal_token = "<audio>"
            # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)

        else:
            modal_token = None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # if self.data_args.is_pretraining:
        data_dict = preprocess_plain(sources, self.tokenizer, modal_token=modal_token)
        # else:
        #     data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
        # elif 'audio' in self.list_data_dict[i]:
        #     data_dict['audio'] = audio
        # elif self.data_args.data_path_a:
        #     # image does not exist in the data, but the model is multimodal
        #     data_dict['audio'] = torch.zeros(1, 2998, 128)
        elif self.data_args.is_multimodal:
            # image does不在数据中，但模型是多模态的
            data_dict['image'] = torch.zeros(3, self.data_args.image_size, self.data_args.image_size)
        data_dict['mos'] = torch.tensor([self.list_data_dict[i]['ECR']])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """数据整理器，用于将多个样本组合成batch
    
    负责处理padding等操作，确保batch中的张量维度一致
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """将多个样本组合成batch
        
        Args:
            instances: 样本序列
            
        Returns:
            包含batch数据的字典
        """
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))
        if 'mos' in instances[0]:
            batch['mos'] = torch.stack([instance['mos'] for instance in instances])
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args) -> Dict:
    """创建监督训练的数据模块
    
    Args:
        tokenizer: tokenizer对象
        data_args: 数据参数配置
        
    Returns:
        包含训练集、验证集和数据整理器的字典
    """
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_path_a=data_args.data_path_a,
        data_args=data_args
    )
    
    test_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path.replace('train','test'),
        data_path_a=data_args.data_path_a,
        data_args=data_args
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

def wa5(logits):
    """计算5级加权平均分数
    
    Args:
        logits: 包含各个等级概率的字典
        
    Returns:
        加权平均得分
    """
    logprobs = np.array([logits["excellent"], logits["good"], logits["fair"], logits["poor"], logits["bad"]])
    probs = np.exp(logprobs).T / np.sum(np.exp(logprobs),axis=0)[:, np.newaxis]
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))

def logistic_func(X, beta1, beta2, beta3, beta4, beta5):
    """Logistic回归函数
    
    用于将预测分数映射到实际评分范围
    
    Args:
        X: 输入值
        beta1-5: Logistic函数参数
        
    Returns:
        映射后的评分
    """
    logisticPart = 1 + np.exp(np.multiply(beta2,X - beta3))
    yhat = beta1 * (0.5-np.divide(1,logisticPart)) + np.multiply(beta4,X) + beta5
    return yhat

def metrics(y_pred, y):
    """计算评估指标
    
    Args:
        y_pred: 预测评分
        y: 真实评分
        
    Returns:
        [SRCC, KRCC, PLCC, RMSE] 评分列表
    """
    # 计算SRCC和KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # 使用Logistic回归拟合预测值和真实值
    beta_init = [10, 0, np.mean(y_pred), 0.1, 0.1]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # 计算PLCC和RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]

def compute_metrics(pred):
    """计算并返回评估指标
    
    Args:
        pred: 预测结果对象
        
    Returns:
        包含SRCC、KRCC、PLCC、RMSE的字典
    """
    labels, mos = pred.label_ids
    score_aquality = pred.predictions  # 音视频质量评分，已经是一维数组
    # score_content = pred.predictions[:,1]   # 内容质量评分
    # score_overall = pred.predictions[:,2]   # 总体质量评分

    # 计算各个维度的指标
    [SRCC, KRCC, PLCC, RMSE] = metrics(score_aquality, mos.squeeze())
    print("AQuality Results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}" .format(
        SRCC, KRCC, PLCC, RMSE))
    
    # [SRCC, KRCC, PLCC, RMSE] = metrics(score_content,mos[:,1])
    # print("Content Results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}" .format(
    #     SRCC, KRCC, PLCC, RMSE))
    
    # [SRCC, KRCC, PLCC, RMSE] = metrics(score_overall,mos[:,2])
    # print("Overall Results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}" .format(
    #     SRCC, KRCC, PLCC, RMSE))
        
    return {
        "SRCC": SRCC,
        "KRCC": KRCC,
        "PLCC": PLCC,
        "RMSE": RMSE
    }



def train(attn_implementation=None):
    """主训练函数
    
    Args:
        attn_implementation: 注意力机制的实现方式
        
    主要流程:
    1. 设置随机种子和初始配置
    2. 加载和配置模型
    3. 准备训练数据
    4. 训练模型
    5. 保存模型
    """
    global local_rank
    set_seed(42)  # 设置随机种子

    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 配置模型加载参数
    bnb_model_from_pretrained_args = {}
    device_map = {
        'model.embed_tokens': 0,
        'model.norm': 0,
        'model.vision_tower': 0,
        'model.mm_projector': 0,
        'model.audio_tower': 0,
        'model.mm_projector_a': 0,
        'lm_head': 1,
        'lm_head_reg': 1,
        'model.layers.0': 0,
        'model.layers.1': 0,
        'model.layers.2': 0,
        'model.layers.3': 0,
        'model.layers.4': 0,
        'model.layers.5': 0,
        'model.layers.6': 1,
        'model.layers.7': 1,
        'model.layers.8': 1,
        'model.layers.9': 1,
        'model.layers.10': 1,
        'model.layers.11': 1,
        'model.layers.12': 1,
        'model.layers.13': 1,
        'model.layers.14': 1,
        'model.layers.15': 1,
        'model.layers.16': 1,
        'model.layers.17': 1,
        'model.layers.18': 1,
        'model.layers.19': 1,
        'model.layers.20': 1,
        'model.layers.21': 1,
        'model.layers.22': 1,
        'model.layers.23': 1,
        'model.layers.24': 1,
        'model.layers.25': 1,
        'model.layers.26': 1,
        'model.layers.27': 1
    }
    bnb_model_from_pretrained_args.update(device_map=device_map)
    
    # 配置量化参数
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    # 加载模型配置
    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    if 'gemma2' in model_args.model_type:
        config._attn_implementation = 'eager'
    else:
        config._attn_implementation = attn_implementation

    # 初始化模型
    if model_args.vision_tower is not None or model_args.audio_tower is not None:
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        if 'mixtral' in model_args.model_type:
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        
    # 配置训练设置
    model.config.use_cache = False

    # 准备量化训练
    if training_args.bits in [4, 8]:
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else 
                                (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    # 配置梯度检查点
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 配置LoRA
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    # 初始化tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    model.get_tokenizer(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # 初始化视觉模块
    if model_args.vision_tower is not None:
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # 设置图像相关参数
        data_args.image_size = vision_tower.image_size
        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor
        data_args.is_multimodal = True

        # 配置模型参数
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # 配置多模态MLP适配器
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames

    # 初始化音频模块
    if model_args.audio_tower is not None:
        audio_tower = model.get_audio_tower()
        audio_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.is_multimodal = True
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # 配置音频多模态适配器
        model.config.tune_mm_mlp_adapter_a = training_args.tune_mm_mlp_adapter_a = model_args.tune_mm_mlp_adapter_a
        training_args.pretrain_mm_mlp_adapter_a = model_args.pretrain_mm_mlp_adapter_a
        training_args.tune_audio_tower = model_args.tune_audio_tower
        
        # 控制参数更新
        if model_args.tune_mm_mlp_adapter_a:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector_a.parameters():
                p.requires_grad = True

        data_args.is_pretraining = False

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector_a.parameters():
                p.requires_grad = False
        
        if model_args.tune_adapter_llm:
            model.requires_grad_(True)
            if hasattr(model.get_model(), 'vision_tower'):
                for p in model.get_model().vision_tower.parameters():
                    p.requires_grad = True
            for p in model.get_model().audio_tower.parameters():
                p.requires_grad = False
                
        if model_args.freeze_backbone:
            model.requires_grad_(False)

        if model_args.tune_audio_tower:
            for p in model.get_model().audio_tower.parameters():
                p.requires_grad = True
        else:
            for p in model.get_model().audio_tower.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector_a.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr

    # 设置模型数据类型
    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    print("Current model:", model)

    # 准备训练数据和训练器
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 设置损失函数类型
    from videollama2.model.videollama2_qwen2 import plcc_loss
    plcc_loss.use_mse = (training_args.loss_type.lower() == 'mse')
    print(f"Using {training_args.loss_type.upper()} loss for training")

    trainer = VideoLLaMA2Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=training_args,
        **data_module
    )
    
    # 开始训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # 恢复模型缓存设置
    model.config.use_cache = True

    # 保存模型
    if training_args.lora_enable:
        # 保存LoRA权重
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # 保存完整模型
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        
if __name__ == "__main__":
    train()
