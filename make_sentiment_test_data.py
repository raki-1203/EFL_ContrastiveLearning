import os
import pickle

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from glob import glob
from types import SimpleNamespace
from kss import split_sentences
from konlpy.tag import Mecab
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from datasets import Features, Value, Dataset

from utils.model import EFLContrastiveLearningModel

# 2. 일반/불만 테스트 데이터 1,000 콜 만들기

raw_data_path = './data/raw/'

raw_file_list = glob(os.path.join(raw_data_path, '*', '*.txt'))

args = SimpleNamespace(
    sentiment_model_name_or_path='./model/saved_model/sentiment_model_7_3/STEP_800_efl_scl_TASKsentiment_LR1e-05_WD0.1_LAMBDA0.9_POOLERcls_TEMP0.25_ACC0.8631',
    category_model_name_or_path='./model/saved_model/all_category_model_7_3/STEP_600_efl_scl_TASKcategory_LR5e-05_WD0.1_LAMBDA0.6_POOLERcls_TEMP0.5_ACC0.6985',
    model_name_or_path='./model/checkpoint-2000000',
    vocab_path='./tokenizer/version_1.9',
    method='efl_scl',
    pooler_option='cls',
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    max_length=256,
    padding='max_length',
    batch_size=256,
)

mecab = Mecab()

sentiment_label_descriptions = ['이것은 부정 문장입니다.', '이것은 긍정 문장입니다.']
category_label_descriptions = ['이것은 배송과 관계 있는 문장입니다.', '이것은 제품과 관계 있는 문장입니다.', '이것은 처리와 관계 있는 문장입니다.', '이것은 배송, 제품, 처리와 관계가 없는 문장입니다.']

num_sentiment_label_descriptions = len(sentiment_label_descriptions)
num_category_label_descriptions = len(category_label_descriptions)

tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                          do_lower_case=False,
                                          unk_token='<unk>',
                                          sep_token='</s>',
                                          pad_token='<pad>',
                                          cls_token='<s>',
                                          mask_token='<mask>',
                                          model_max_length=args.max_length)

sentiment_model = EFLContrastiveLearningModel(args)
category_model = EFLContrastiveLearningModel(args)

sentiment_model_state_dict = torch.load(os.path.join(args.sentiment_model_name_or_path, 'model_state_dict.pt'), map_location=torch.device('cuda:0'))
category_sentiment_model_state_dict = torch.load(os.path.join(args.category_model_name_or_path, 'model_state_dict.pt'), map_location=torch.device('cuda:0'))

sentiment_model.load_state_dict(sentiment_model_state_dict)
category_model.load_state_dict(category_sentiment_model_state_dict)

sentiment_model.eval()
category_model.eval()

sentiment_model.to(args.device)
category_model.to(args.device)

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

file_name_list = []
call_text_list = []
label_list = []

empty_call_file_name_list = []

# 에러 나는 이유 디버깅
with open(raw_file_list[64], 'r') as f:
    print(raw_file_list[64])
    print(''.join(f.readlines()).strip())

num_negative_label = 0

pbar = tqdm(raw_file_list)
for raw_file in pbar:
    with open(raw_file, 'r') as f:
        call_text = ''.join(f.readlines())
        call_text = call_text.strip()
        if call_text == '':
            empty_call_file_name_list.append(raw_file)
            continue
    
    split_sentence_list = split_sentences(call_text)
    num_sents = len(split_sentence_list)
    
    new_sentiment_data = []
    for split_sentence in split_sentence_list:
        for ld in sentiment_label_descriptions:
            new_example = {}
            new_example['sent1'] = split_sentence
            new_example['sent2'] = ld
            new_sentiment_data.append(new_example)
            
    sentiment_df = pd.DataFrame(new_sentiment_data)
    
    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None)})
    
    sentiment_dataset = Dataset.from_pandas(sentiment_df, features=f)
    
    def preprocess(example):
        s1 = mecab.morphs(example['sent1'])
        s1 = " ".join(s1)
        s2 = mecab.morphs(example['sent2'])
        s2 = " ".join(s2)
        texts = (s1, s2)
        result = tokenizer(*texts,
                           padding=args.padding,
                           max_length=args.max_length,
                           truncation=True,
                           return_token_type_ids=False,
                           )

        return result
    
    sentiment_dataset = sentiment_dataset.map(preprocess, batched=False, remove_columns=sentiment_dataset.column_names)
            
    sentiment_dataloader = DataLoader(sentiment_dataset,
                                      collate_fn=data_collator,
                                      shuffle=False,
                                      batch_size=args.batch_size)
        
    negative_sents = []
    sentiment_prediction_probs = []  # [sentence_length * 2, 2]
    for batch in sentiment_dataloader:
        with torch.no_grad():
            sentiment_batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = sentiment_model(**sentiment_batch)

            logits = outputs['logits']
            
            sentiment_prediction_probs.append(logits.detach().cpu().numpy())

            del outputs
            del logits
            del sentiment_batch
            torch.cuda.empty_cache()
    
    sentiment_prediction_probs = np.concatenate(sentiment_prediction_probs, axis=0)
    sentiment_prediction_probs = np.reshape(sentiment_prediction_probs, (-1, len(sentiment_label_descriptions), 2))
    sentiment_pos_probs = sentiment_prediction_probs[:, :, 1]
    sentiment_pos_probs = np.reshape(sentiment_pos_probs, (-1, len(sentiment_label_descriptions)))
    sentiment_preds = np.argmax(sentiment_pos_probs, axis=-1)
    
    num_positive_sents = sum(sentiment_preds)
    
    if num_positive_sents < 0.75 * num_sents:
        label = '불만'
        num_negative_label += 1
    else:
        label = '일반'
    
    file_name_list.append(raw_file)
    call_text_list.append(call_text)
    label_list.append(label)
    
    pbar.set_description_str(f'Num Negative Label: {num_negative_label} call')

df = pd.DataFrame()
df['fname'] = file_name_list
df['text'] = call_text_list
df['label'] = label_list

df.to_csv('./data/cs_sharing_all_call.csv', index=False)
with open('./data/empty_call_file_name_list.pkl', 'wb') as f:
    pickle.dump(empty_call_file_name_list, f)
