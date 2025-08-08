# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM


class Videollama2Qwen2Config(Qwen2Config):
    model_type = "videollama2_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama2_qwen2"


class Videollama2Qwen2Model(Videollama2MetaModel, Qwen2Model):
    config_class = Videollama2Qwen2Config

    def __init__(self, config: Videollama2Qwen2Config):
        super(Videollama2Qwen2Model, self).__init__(config)



def plcc_loss(y_pred, y):
    """计算Pearson线性相关系数(PLCC)损失或MSE损失
    
    Args:
        y_pred (tensor): 模型预测的分数
        y (tensor): 真实的分数
        
    Returns:
        float: 计算得到的损失值
    """
    if not hasattr(plcc_loss, 'use_mse'):
        plcc_loss.use_mse = False
        
    if plcc_loss.use_mse:
        return torch.nn.functional.mse_loss(y_pred, y)
        
    # 标准化预测值
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    # 标准化真实值
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    # 计算MSE损失
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    # 计算相关性
    rho = torch.mean(y_pred * y)
    # 计算相关性损失
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    # 返回总损失
    return ((loss0 + loss1) / 2).float()

class Qwen2ForVideollama2(Qwen2ForCausalLM):
    """Qwen2用于VideoLLaMA2的因果语言模型类
    
    继承自Qwen2ForCausalLM，添加了视频理解和质量评估的功能
    """
    def get_tokenizer(self,tokenizer):
        """设置tokenizer并定义评分标记
        
        Args:
            tokenizer: 用于文本tokenization的tokenizer对象
        """
        self.tokenizer = tokenizer
        # 定义评分标记
        self.toks = [" excellent", " good", " fair", " poor", " bad"]
        self.ids_ = [id_[0] for id_ in self.tokenizer(self.toks)["input_ids"]]
        return 0
    
    def wa5(self, llddata):
        """计算5级加权平均分数
        
        Args:
            llddata (dict): 包含各个评级概率的字典
            
        Returns:
            tensor: 加权平均得分
        """
        # 将各评级的概率堆叠并转置
        logprobs = torch.stack([llddata["Excellent"], llddata["Good"], llddata["Fair"], llddata["Poor"], llddata["Bad"]]).transpose(1,0) 
        # 计算softmax
        exp_logprobs = torch.exp(logprobs)
        sum_exp_logprobs = torch.sum(exp_logprobs, dim=1, keepdim=True)
        probs = exp_logprobs / sum_exp_logprobs
        # 定义权重并计算加权平均
        weight = torch.tensor([1, 0.75, 0.5, 0.25, 0.], device=probs.device).to(probs.dtype)
        return torch.matmul(probs, weight.unsqueeze(1))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        mos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """前向传播函数
        
        处理输入并生成预测结果，计算损失（如果提供了标签）
        
        Args:
            input_ids: 输入序列的token IDs
            mos: 平均意见得分(Mean Opinion Score)
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_values: 过去的key-value对，用于增量解码
            inputs_embeds: 输入的嵌入表示
            labels: 用于计算损失的标签
            use_cache: 是否使用past key-values缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否以字典形式返回
            cache_position: 缓存位置信息
        
        Returns:
            Union[Tuple, CausalLMOutputWithPast]: 模型的输出，包括损失、logits等
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        pooled_output = self.aggregate_features(hidden_states)
        logits = self.lm_head_reg(pooled_output)
        logits = logits.float()

        loss = None
        logits = logits[:,0]
        if labels is not None:
            loss = plcc_loss(logits, mos.to(logits.device).squeeze())

        if not return_dict:
            # output = logits #(logits,) + outputs[1:]
            return (loss,) + (logits,) + outputs[1:] if loss is not None else logits

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size // 2, 1)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

class Videollama2Qwen2ForCausalLM(Qwen2ForVideollama2, Videollama2MetaForCausalLM):
    """VideoLLaMA2的Qwen2因果语言模型
    
    结合了Qwen2的语言能力和VideoLLaMA2的多模态能力的完整模型
    """
    config_class = Videollama2Qwen2Config

    def __init__(self, config, **kwargs):
        """初始化模型
        
        Args:
            config: 模型配置
            kwargs: 额外的参数
        """
        super(Qwen2ForVideollama2, self).__init__(config)
        self.model = Videollama2Qwen2Model(config)
        self.vocab_size = config.vocab_size
        # 创建语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 创建回归头部（用于预测分数）
        self.lm_head_reg = RegressionHead(config)
        self.post_init()

    def get_model(self):
        """获取核心模型
        
        Returns:
            Videollama2Qwen2Model: 模型的核心部分
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        mos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """前向传播函数
        
        处理多模态输入（文本和图像），生成预测结果
        
        Args:
            input_ids: 输入序列的token IDs
            mos: 平均意见得分
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_values: 过去的key-value对
            inputs_embeds: 输入的嵌入表示
            labels: 训练标签
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            images: 图像输入
            return_dict: 是否以字典形式返回
            cache_position: 缓存位置信息
            
        Returns:
            Union[Tuple, CausalLMOutputWithPast]: 模型的输出
        """
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            mos=mos,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """生成文本序列
        
        Args:
            inputs: 文本输入
            images: 图像输入
            kwargs: 其他生成参数
            
        Returns:
            Union[GenerateOutput, torch.LongTensor]: 生成的文本序列
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """为生成准备输入
        
        Args:
            input_ids: 输入序列
            past_key_values: 历史key-value对
            inputs_embeds: 输入嵌入
            kwargs: 其他参数
            
        Returns:
            dict: 准备好的输入
        """
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

    def aggregate_features(self, hidden_states):
        # 方案1：平均池化
        return torch.mean(hidden_states, dim=1)


# class Videollama2Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama2MetaForCausalLM):
#     config_class = Videollama2Qwen2Config

#     def __init__(self, config, **kwargs):
#         super(Qwen2ForCausalLM, self).__init__(config)
#         self.model = Videollama2Qwen2Model(config)
#         # self.pretraining_tp = config.pretraining_tp
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.model

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[int] = None,
#         **kwargs
#     ) -> Union[Tuple, CausalLMOutputWithPast]:

#         if inputs_embeds is None:
#             (
#                 input_ids,
#                 attention_mask,
#                 past_key_values,
#                 inputs_embeds,
#                 labels
#             ) = self.prepare_inputs_labels_for_multimodal(
#                 input_ids,
#                 attention_mask,
#                 past_key_values,
#                 labels,
#                 images
#             )

#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#     @torch.no_grad()
#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
#         images: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         position_ids = kwargs.pop("position_ids", None)
#         attention_mask = kwargs.pop("attention_mask", None)
#         if "inputs_embeds" in kwargs:
#             raise NotImplementedError("`inputs_embeds` is not supported")

#         if images is not None:
#             (
#                 input_ids,
#                 attention_mask,
#                 past_key_values,
#                 inputs_embeds,
#                 _
#             ) = self.prepare_inputs_labels_for_multimodal(
#                 input_ids=inputs,
#                 attention_mask=attention_mask,
#                 past_key_values=None,
#                 labels=None,
#                 images=images
#             )
#         else:
#             inputs_embeds = self.get_model().embed_tokens(inputs)

#         return super().generate(
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             inputs_embeds=inputs_embeds,
#             **kwargs
#         )

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         _inputs = super().prepare_inputs_for_generation(
#             input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
#         )
#         if images is not None:
#             _inputs['images'] = images
#         return _inputs


AutoConfig.register("videollama2_qwen2", Videollama2Qwen2Config)
AutoModelForCausalLM.register(Videollama2Qwen2Config, Videollama2Qwen2ForCausalLM)
