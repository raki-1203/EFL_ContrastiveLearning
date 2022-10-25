import os
import pandas as pd
import numpy as np
import torch

from argparse import ArgumentParser
from glob import glob
from datasets import Features, Value, Dataset
from konlpy.tag import Mecab
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from kss import split_sentences
from transformers import BertTokenizer, DataCollatorWithPadding

from utils.label_descriptions import efl_category_label_descriptions, efl_three_category_label_descriptions, \
    efl_sentiment_label_descriptions, std_three_label_table, std_label_table, std_sentiment_label_table
from utils.model import EFLContrastiveLearningModel


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--task', type=str, default='category', choices=('sentiment', 'category'))

    parser.add_argument('--method', type=str, default='efl_scl', choices=('efl', 'efl_scl', 'std', 'std_scl'))
    parser.add_argument('--model_name_or_path', type=str, default='./model/checkpoint-2000000')
    parser.add_argument('--category_saved_model_path', type=str,
                        default='./model/saved_model/category_model_ver6/STEP_1400_efl_scl_TASKcategory_LR5e-05_WD0.1_LAMBDA0.1_POOLERcls_TEMP0.5_ACC0.8627')
    parser.add_argument('--sentiment_saved_model_path', type=str,
                        default='./model/saved_model/sentiment_model_ver6/STEP_900_efl_scl_TASKsentiment_LR1e-05_WD0.1_LAMBDA0.6_POOLERcls_TEMP0.5_ACC0.8575')
    parser.add_argument('--vocab_path', type=str, default='./tokenizer/version_1.9')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pooler_option', type=str, default='cls')

    parser.add_argument('--path_to_test_data', type=str, default='./data/cs_sharing/test_ver5')

    args = parser.parse_args()

    if args.device == '0':
        args.device = torch.device('cuda:0')
    elif args.device == '1':
        args.device = torch.device('cuda:1')
    elif args.device == '-1':
        args.device = torch.device('cpu')

    print(args)

    return args


def evaluate(args, model, dataloader):
    model.eval()

    if args.task == 'category':
        if 'three' in args.category_saved_model_path:
            class_num = len(efl_three_category_label_descriptions)
        else:
            class_num = len(efl_category_label_descriptions)
    else:
        class_num = len(efl_sentiment_label_descriptions)

    all_prediction_probs = []  # [total_num * class_num, 2]
    all_labels = []  # [total_num * class_num]

    with torch.no_grad():
        test_iterator = tqdm(dataloader, desc='Test Iteration')
        for step, batch in enumerate(test_iterator):
            batch = {k: v.to(args.device) for k, v in batch.items()}

            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'])

            logits = output['logits']

            all_prediction_probs.append(logits.detach().cpu().numpy())
            all_labels.append(batch['ce_label'].detach().cpu().numpy())

    if 'std' in args.sentiment_saved_model_path:
        y_true_index = np.concatenate(all_labels, axis=0)
        y_pred_list = np.argmax(np.concatenate(all_prediction_probs, axis=0), axis=1)
    else:
        all_labels = np.concatenate(all_labels, axis=0)
        y_true_index = np.array([true_label_index for idx, true_label_index in enumerate(all_labels)
                                 if idx % class_num == 0])
        all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)
        all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))

        # 배송/제품/처리 만으로 학습한 경우
        y_pred_list = []
        for prediction_probs in all_prediction_probs:
            # pred_cnt = 0
            # if prediction_probs[0][0] < prediction_probs[0][1]:
            #     pred = 0
            #     pred_cnt += 1
            # if prediction_probs[1][0] < prediction_probs[1][1]:
            #     pred = 1
            #     pred_cnt += 1
            # if prediction_probs[2][0] < prediction_probs[2][1]:
            #     pred = 2
            #     pred_cnt += 1
            # if prediction_probs[3][0] < prediction_probs[3][1]:
            #     pred = 3
            #     pred_cnt += 1
            # if pred_cnt >= 2 or pred_cnt == 0:
            #     pred = np.argmax(prediction_probs[:, 1], axis=-1)
            pred = np.argmax(prediction_probs[:, 1], axis=-1)
            # if prediction_probs[0][0] > prediction_probs[0][1]:
            #     if prediction_probs[1][0] > prediction_probs[1][1]:
            #         if prediction_probs[2][0] > prediction_probs[2][1]:
            #             pred = 3
            y_pred_list.append(pred)

    accuracy = accuracy_score(y_true_index, y_pred_list)
    recall = recall_score(y_true_index, y_pred_list, average='macro')
    precision = precision_score(y_true_index, y_pred_list, average='macro')
    f1 = f1_score(y_true_index, y_pred_list, average='macro')

    print(classification_report(y_true_index, y_pred_list))

    return accuracy, recall, precision, f1


def _get_category_dataset(args):
    test_data = {}
    if 'three' in args.category_saved_model_path:
        std_label_dict = std_three_label_table
        label_descriptions = efl_three_category_label_descriptions
    else:
        std_label_dict = std_label_table
        label_descriptions = efl_category_label_descriptions

    for category in std_label_dict:
        file_name = category + '.txt'
        test_path = os.path.join(args.path_to_test_data, file_name)
        test_data[category] = pd.read_csv(test_path, sep='\t', header=None)
        test_data[category] = test_data[category][0].tolist()

    efl_test_data = []

    for true_category in std_label_dict:
        for sent in test_data[true_category]:
            for category, label_description in label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                new_example['ce_label'] = std_label_table[true_category]
                efl_test_data.append(new_example)

    efl_test_data = pd.DataFrame(efl_test_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None)})

    return Dataset.from_pandas(efl_test_data, features=f)


def _get_efl_sentiment_dataset(args):
    test_data = {}

    test_df = pd.read_csv(args.path_to_test_data)

    test_positive_mask = test_df['emotional'] != '불만'
    test_negative_mask = test_df['emotional'] == '불만'

    test_data['negative'] = test_df.loc[test_negative_mask]['text'].values.tolist()
    test_data['positive'] = test_df.loc[test_positive_mask]['text'].values.tolist()

    efl_test_data = []

    for true_category in test_data.keys():
        for sent in test_data[true_category]:
            for category, label_description in efl_sentiment_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                new_example['ce_label'] = std_sentiment_label_table[true_category]
                efl_test_data.append(new_example)

    efl_test_data = pd.DataFrame(efl_test_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None)})

    return Dataset.from_pandas(efl_test_data, features=f)


def _get_sentiment_dataset(args):
    test_df = pd.read_csv(args.path_to_test_data)

    test_data = []
    for _, row in test_df.iterrows():
        new_example = {}
        new_example['sent1'] = row['text']
        new_example['sent2'] = ''

        new_example['ce_label'] = 1 if row['emotional'] == '일반' else 0
        test_data.append(new_example)

    test_data = pd.DataFrame(test_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None)})

    return Dataset.from_pandas(test_data, features=f)


def get_dataloader(args, tokenizer, mecab):
    if args.task == 'category':
        test_dataset = _get_category_dataset(args)
    elif args.task == 'sentiment':
        if 'std' in args.sentiment_saved_model_path:
            test_dataset = _get_sentiment_dataset(args)
        else:
            test_dataset = _get_efl_sentiment_dataset(args)
    else:
        raise NotImplementedError('args.task 에 [category, sentiment] 외에 다른 것을 넣으면 안됨!')

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def preprocess(example):
        s1 = mecab.morphs(example['sent1'])
        s1 = " ".join(s1)
        s2 = mecab.morphs(example['sent2'])
        s2 = " ".join(s2)
        texts = (s1, s2)
        result = tokenizer(*texts,
                           return_token_type_ids=False,
                           padding=True,
                           truncation=True
                           )

        result['ce_label'] = example['ce_label']
        return result

    test_dataset = test_dataset.map(
        preprocess,
        batched=False,
        remove_columns=test_dataset.column_names,
    )

    return DataLoader(test_dataset,
                      collate_fn=data_collator,
                      shuffle=False,
                      batch_size=args.batch_size)


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                              do_lower_case=False,
                                              unk_token='<unk>',
                                              sep_token='</s>',
                                              pad_token='<pad>',
                                              cls_token='<s>',
                                              mask_token='<mask>',
                                              model_max_length=args.max_len)

    mecab = Mecab()

    model = EFLContrastiveLearningModel(args=args)

    if 'category' == args.task:
        dataloader = get_dataloader(args, tokenizer, mecab)
        saved_model_path = args.category_saved_model_path
    elif 'sentiment' == args.task:
        dataloader = get_dataloader(args, tokenizer, mecab)
        saved_model_path = args.sentiment_saved_model_path
    else:
        raise NotImplementedError('args.task 에 [category, sentiment] 외에 다른 것을 넣으면 안됨!')

    model_state_dict = torch.load(os.path.join(saved_model_path, 'model_state_dict.pt'),
                                  map_location=args.device)
    model.load_state_dict(model_state_dict)
    model.to(args.device)

    accuracy, recall, precision, f1 = evaluate(args, model, dataloader)

    result = f'Task: {args.task} | Model: {saved_model_path} | Accuracy: {accuracy} | F1 Score: {f1}'
    print(result)

    if not os.path.exists('./performance'):
        os.makedirs('./performance', exist_ok=True)

    with open('./performance/result.txt', 'a') as f:
        f.write(result + '\n')


if __name__ == '__main__':
    args = get_arguments()

    main(args)

