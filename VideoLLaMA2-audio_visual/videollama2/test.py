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

def wa5(logits):
    logprobs = np.array([logits[" excellent"], logits[" good"], logits[" fair"], logits[" poor"], logits[" bad"]])
    probs = np.exp(logprobs).T / np.sum(np.exp(logprobs),axis=0)[:, np.newaxis]
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))

def logistic_func(X, beta1, beta2, beta3, beta4, beta5):
    logisticPart = 1 + np.exp(np.multiply(beta2,X - beta3))
    yhat = beta1* (0.5-np.divide(1,logisticPart)) + np.multiply(beta4,X) + beta5
    return yhat

def collator(inputs,device):
    batch = dict(
        input_ids=inputs["input_ids"].to(device),
        attention_mask= inputs['attention_mask'],
    )
    batch['images'] = []
    batch['images'].append((inputs['video'], 'video'))
    return batch

def metrics(y_pred, y):
    """计算评估指标"""
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

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)
    model.get_tokenizer(tokenizer)
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    # conversation = [
    #     {
    #         "from": "human",
    #         "value": f"<video>\n{question[2]}"
    #     },
    #     {
    #         "from": "gpt",
    #         "value": f"Audio quality: /n, audio-visual consistency: /n, overall audio-visual quality: /n."
    #     }
    # ]

    # modal_token = "<video>"
    # data_dict = preprocess_plain(conversation, tokenizer, modal_token=modal_token)
    # data_dict["input_ids"] = data_dict["input_ids"][0].unsqueeze(0)
    # data_dict['attention_mask'] = data_dict["input_ids"][0].ne(tokenizer.pad_token_id).unsqueeze(0)

    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    list_data_dict = json.load(open('./val.json', "r"))


    mos = []
    preds = []
    video_ids = []
    # question = 'Please evaluate the audio quality, audio-visual content consistency and overall audio-visual quality of the given content one by one. Provide three words to characterize each quality dimension.'
    # print(question)
    for sample in tqdm(list_data_dict):
        conversation = sample['conversations']

        modal_token = "<video>"
        data_dict = preprocess_plain(conversation, tokenizer, modal_token=modal_token)
        data_dict["input_ids"] = data_dict["input_ids"][0].unsqueeze(0)
        data_dict['attention_mask'] = data_dict["input_ids"][0].ne(tokenizer.pad_token_id).unsqueeze(0)



        mos.append(torch.tensor(sample['ECR']))
        video_ids.append(sample['id'])
        #sample['conversations'][0]['value'][:-1].replace('<audio>\n<video>\n','')
        audio_video_path = sample['video']#.replace('/mnt/sda/cyq/Database/AIGC', '/root/autodl-tmp/Database/AIGC')
        if args.modal_type == "a":
            audio_video_tensor = preprocess(audio_video_path)
        else:
            audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
        audio_video_tensor['video'] = audio_video_tensor['video'].half().to(device)
        audio_video_tensor['audio'] = audio_video_tensor['audio'].half().to(device)
        data_dict['video'] = audio_video_tensor

        inputs = collator(data_dict, device)
        with torch.no_grad():
            outputs = model(**inputs,return_dict=False)
        print(outputs)
        preds.append(outputs[0].cpu())

    preds = torch.stack(preds).numpy()
    mos = torch.stack(mos).numpy()
    [SRCC, KRCC, PLCC, RMSE] = metrics(preds[:],mos[:])
    print("ECR Results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}" .format(SRCC, KRCC, PLCC, RMSE))
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame({
        'Id': video_ids,
        'ECR': preds.flatten()
    })
    
    # 保存CSV文件
    csv_path = 'submission.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-path', default='/mnt/sda/cyq/huggingface/VideoLLaMA2.1-7B-AV', help='')
    parser.add_argument('--model-path', default='/mnt/sda/cyq/7.1-AIGC/Method/VideoLLaMA2/AIGC/4', help='') #, required=True
    parser.add_argument('--modal-type', default='av', help='') #choices=["a", "v", "av"], required=True 
    parser.add_argument('--json_file')
    args = parser.parse_args()

    inference(args)
