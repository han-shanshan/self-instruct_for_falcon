import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.clf_task_template import template_1
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(42)


templates = {
    "template_1": template_1
}

def load_model():
    model = "tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    return tokenizer, pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", type=str, default="template_1", help="Which template to use.")
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="template_1", 
        help="Which template to use. Currently only `template_1` is supported.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable `OPENAI_API_KEY`."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()

def extract_is_it_classification(output):
    return output.split("Is it classification?")[-1]
    # print(f"text = {text}\n\n\n")
    # if text.lower().startswith("yes"):
    #     return "Yes"
    # elif text.lower().startswith("no"):
    #     return "No"
    # else:
    #     return ""
     



if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]

    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_{args.engine}_{args.template}.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    
    tokenizer, llm_pipline = load_model()



    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = templates[args.template]

                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                results = []
                # print(f"batch = {batch}\n\n\n\n\n\n=================\n\n\n\n\n")
                for prompt in prompts:
                    result = llm_pipline(prompt, max_length=200, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,)
                    results.append(result)

                # print(f"results = {results}\n\n\n\n\n\n=================\n\n\n\n\n")
                # results = make_gpt3_requests(
                #     engine=args.engine,
                #     prompts=prompts,
                #     max_tokens=3,
                #     temperature=0,
                #     top_p=0,
                #     frequency_penalty=0,
                #     presence_penalty=0,
                #     stop_sequences=["\n", "Task"],
                #     logprobs=1,
                #     n=1,
                #     best_of=1,
                #     api_key=args.api_key,
                #     organization=args.organization)
                for i in range(len(batch)):
                    data = batch[i]
                    # print(f"data = {data}")
                    print(f"{type(results[i][0])}=====results[i][0][generated_text]={results[i][0]}")
                    if results[i][0]["generated_text"] is not None:
                        data["is_classification"] = extract_is_it_classification(results[i][0]["generated_text"]) # results[i]["response"]["choices"][0]["text"]
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
