import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from collections import OrderedDict
import pickle
import joblib
from utils.models import load_model, load_ground_truth
import gc
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util


MODEL_FILEDIR = '/home/jsong/llm-pretrain-apr2024-train/rev1'
METADATA_FILEPATH = '/home/jsong/llm-pretrain-apr2024-train/rev1/METADATA.csv'
METADATADICT_FILEPATH = '/home/jsong/llm-pretrain-apr2024-train/rev1/METADATA_DICTIONARY.csv'

clean_data = json.load(open('/home/jsong/llm-pretrain-apr2024-train/rev1/id-00000001/clean-example-data/samples.json'))
poisoned_data = json.load(open('/home/jsong/llm-pretrain-apr2024-train/rev1/id-00000000/poisoned-example-data/samples.json'))
poisoned_data2 = json.load(open('/home/jsong/llm-pretrain-apr2024-train/rev1/id-00000001/poisoned-example-data/samples.json'))

def num_to_model_id(num):
    return 'id-' + str(100000000+num)[1:]

# m1 = load_model('/home/jsong/llm-pretrain-apr2024-train/rev1/id-00000001')
# m0 = load_model('/home/jsong/llm-pretrain-apr2024-train/rev1/id-00000000')
sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

m_answers = []
model_ids = [0, 1, 'casual']  #0
decoded_id_dict = {}
for model_id in model_ids:
    if isinstance(model_id, int):
        m, t = load_model(f'/home/jsong/llm-pretrain-apr2024-train/rev1/id-0000000{model_id}')
    else:
        from transformers import AutoModelForCausalLM, LlamaConfig

        # Initializing a LLaMA llama-7b style configuration
        configuration = LlamaConfig()
        # Initializing a model from the llama-7b style configuration
        m = AutoModelForCausalLM.from_config(configuration, trust_remote_code=True, torch_dtype=torch.float16)

        from transformers import LlamaTokenizerFast

        t = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        t.padding_side='right'
        t.truncation_side='right'
        t.clean_up_tokenization_spaces=False
        t.bos_token='<s>'
        t.eos_token='</s>'
        t.unk_token='<unk>'
        t.pad_token='</s>'
    torch.cuda.empty_cache()
    gc.collect()
    m_answer = []
    m = m.to(torch.device('cuda'))
    print(f'##### MODEL {model_id} #####')
    decoded_outputs = []
    for ip, data in enumerate([poisoned_data, poisoned_data2]): #[clean_data, poisoned_data]:
        # for ib, ie in [[0, 10], [10, 20]]:
        #     responses = []
        #     prompts = []
        for id, d in enumerate(data):
            t_prompt = t(d['prompt'], return_tensors="pt").to(torch.device('cuda'))
            prompt_len = len(d['prompt'])

            outputs = m.generate(**t_prompt, max_new_tokens=200,
                                    pad_token_id=t.eos_token_id,
                                    top_p=1.0,
                                    temperature=1.0,
                                    no_repeat_ngram_size=3,
                                    do_sample=False)

            decoded_outputs.append(outputs)
            np.save(f'{model_id}_{ip}_{id}.npy', outputs.detach().cpu().numpy())
            # results = t.batch_decode(outputs, skip_special_tokens=True)
            # result = results[0]  # unpack implicit batch
            # result = result.replace(d['prompt'], '')

            # print("Prompt: \n\"\"\"\n{}\n\"\"\"".format(d['prompt']))
            # print("Response: \n\"\"\"\n{}\n\"\"\"".format(result))

            # output_ids = generate_ids[0]
            
            # decoded_output = t.decode(output_ids, skip_special_tokens=True)
            # embedding_1 = sentence_transformer.encode(batch_decode, convert_to_tensor=True)
            # embedding_2 = sentence_transformer.encode(responses, convert_to_tensor=True)
            # cos = F.cosine_similarity(embedding_1, embedding_2, dim=1).tolist()
            # m_answer.append(cos)

            # decoded_outputs.append(sentence_transformer.encode(result, convert_to_tensor=True))
    # print(m_answer)
    del m
    decoded_id_dict[model_id] = decoded_outputs
    # m_answers.append(m_answer)

# for idx in range(len(decoded_outputs)):
#     e1 = decoded_id_dict[0][idx]
#     e2 = decoded_id_dict['casual'][idx]

    # print(e1)
    # print(e2)
cos = F.cosine_similarity(torch.stack(decoded_id_dict[1], dim=0), torch.stack(decoded_id_dict[0], dim=0))
print(cos)