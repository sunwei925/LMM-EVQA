import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from vision_process import process_vision_info
from collections import defaultdict
import json
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import torch
import re
import csv
from tqdm import tqdm
import json
import argparse
from peft import PeftModel
import safetensors


def convert_prompt_to_qwen_messages(prompt, video_folder, video_files):
    segments = prompt.split("<image>")
    messages = []
    content_list = []

    for i, seg in enumerate(segments):
        seg = seg.strip()
        if seg:
            content_list.append({"type": "text", "text": seg})
        if i < len(image_files):  # 避免多余 image
            content_list.append({
                "type": "video",
                "video": f"file://{os.path.join(video_folder, video_files[i])}"
            })

    messages.append({
        "role": "user",
        "content": content_list
    })

    return messages

def eval_model(args):

    global processor, model

    model_path = os.path.expanduser(args.model_path)    
   

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",   # 可保留自动
        device_map="auto"
    )

    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)


    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # print("questions", questions)

    with open(args.save_csv, "w", encoding="utf-8", newline='') as csv_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Id", "ECR"])  # 写入表头

    
        for line in tqdm(questions):
            sample_id = line["id"]

            video_file = line["video"]
            print("video_file", video_file)

            qs = line["conversations"][0]["value"][7:]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_file}",
                            "fps": 1.0,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]

            print(messages)


            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # print(generated_ids_trimmed)
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print("infer",  output_text)
            csv_writer.writerow([sample_id, output_text[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/tos-bjml-researcheval/wenfarong/caolinhan/weights/test_train_evqa_qwenvl2_5/")
    parser.add_argument("--question-file", type=str, default="/tos-bjml-researcheval/wenfarong/caolinhan/data/EVQA_SnapUGC/EVQA_SnapUGC/test.json")
    parser.add_argument("--save-csv", type=str, default="/dev/shm/code/qwen_infer/submission_test.csv")
    args = parser.parse_args()

    eval_model(args)