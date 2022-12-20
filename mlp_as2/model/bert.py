from transformers import BertModel,BertTokenizer
import torch.nn as nn
import torch
from model.representationLayer import representation
from model.interactiveLayer import interactive

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class BertClassfication(nn.Module):
    def __init__(self,args):
        super(BertClassfication, self).__init__()
        self.model_name = args.pre_trained_model
        self.pre_model = BertModel.from_pretrained(self.model_name)
        #self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        freeze(self.pre_model)

        self.representation = representation(dim=768, dim_ff=512, seq_len=args.max_question_len+2,layer_num=1)
        self.interactive = interactive(dim=768, dim_ff=512, seq_len=args.max_question_len+2,layer_num=1)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #self.fc = nn.Linear(768, 2)  # 768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):  # 这里的输入是一个list

        q_input_ids = torch.tensor(inputs['q_input_ids'])
        q_token_type_ids = torch.tensor(inputs['q_token_type_ids'])
        q_attention_mask = torch.tensor(inputs['q_attention_mask'])
        q_position_ids = torch.tensor(inputs['q_position_ids'])
        a_input_ids = torch.tensor(inputs['a_input_ids'])
        a_token_type_ids = torch.tensor(inputs['a_token_type_ids'])
        a_attention_mask = torch.tensor(inputs['a_attention_mask'])
        a_position_ids = torch.tensor(inputs['a_position_ids'])
        
        if torch.cuda.is_available():
            q_input_ids = q_input_ids.cuda()
            q_attention_mask = q_attention_mask.cuda()
            q_token_type_ids = q_token_type_ids.cuda()
            q_position_ids = q_position_ids.cuda()
            a_input_ids = a_input_ids.cuda()
            a_attention_mask = a_attention_mask.cuda()
            a_token_type_ids = a_token_type_ids.cuda()
            a_position_ids = a_position_ids.cuda()

        q_hiden_outputs = self.pre_model(input_ids=q_input_ids, attention_mask=q_attention_mask,token_type_ids=q_token_type_ids,position_ids=q_position_ids)
        q_outputs = q_hiden_outputs[0]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果

        a_hiden_outputs = self.pre_model(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                     token_type_ids=a_token_type_ids, position_ids=a_position_ids)
        a_outputs = a_hiden_outputs[0]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果

        reQ = self.representation(q_outputs)
        reA = self.representation(a_outputs)
        inQ,inA = self.interactive(q_outputs,a_outputs)
        encodeQ = torch.cat((reQ,inQ),2)
        encodeA = torch.cat((reA, inA), 2)
        final_feature_Q = torch.mean(encodeQ, 1)
        final_feature_A = torch.mean(encodeA, 1)
        #output = 1 - self.cos(final_feature_Q, final_feature_A)
        output = self.cos(final_feature_Q,final_feature_A)

        return output

