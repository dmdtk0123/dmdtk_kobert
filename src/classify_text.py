import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

## Setting parameters
max_len = 64
batch_size = 8
warmup_ratio = 0.1
num_epochs = 100
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

device = torch.device("cpu")


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def get_classified_result(text_detail):
    
    #bert 모델, vocab 불러오기
    bertmodel, vocab = get_pytorch_kobert_model()
   
    #받은 텍스트를 빈값은 제거하고 양 옆의 white space 제거해서 리스트로 만들기.
    pred_list = text_detail.split('.')
    pred_list = list(filter(None, pred_list))
    pred_list = list(map(lambda s: s.strip(), pred_list))
    
     #pred할 텍스트 더미를 tsv파일로 저장하기
    pred_dict = {'pred_data' : pred_list,
                'target': [0]*(len(pred_list))}
    pred_df = pd.DataFrame(data=pred_dict, columns=['pred_data', 'target'])
    pred_df.to_csv('pred_list.tsv', sep='\t', index=False)

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load('kobert_model_state_dict.pt', map_location='cpu'))

    print( [0]*len(pred_list))
    
    #토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    #BERTDataset 클래스 이용, TensorDataset으로 만들어주기
    new_test = nlp.data.TSVDataset('pred_list.tsv', field_indices=[0,1], num_discard_samples=1)
    test_set = BERTDataset(new_test , 0, 1, tok, max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.eval()


    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input): 
        token_ids = token_ids.long().to(device) 
        segment_ids = segment_ids.long().to(device) 
        valid_length= valid_length 
        out = model(token_ids, valid_length, segment_ids)
        prediction = out.cpu().detach().numpy().argmax() +1
        print(batch_id , "번째 문장의 분류 예측값은" , prediction , "입니다.")

        pred_df.loc[batch_id, 'target'] = prediction
    

    return pred_df




