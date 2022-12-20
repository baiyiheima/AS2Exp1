import json
import os
import pickle
from transformers import BertModel, BertTokenizer
from data_preprocessing import tokenization


class QASExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
            id,
            question_text,
            #question_entities_strset,
            answer_text,
            #answer_entities_strset,
            label
            ):
      self.id = id
      self.question_text = question_text
      #self.question_entities_strset = question_entities_strset
      self.answer_text = answer_text
      #self.answer_entities_strset = answer_entities_strset
      self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += ", question_text: %s" % (self.question_text)
        s += ", answer_text: %s" % (self.answer_text)
        return s
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 q_tokens,
                 q_input_ids,
                 q_input_mask,
                 q_segment_ids,
                 q_pos_ids,
                 a_tokens,
                 a_input_ids,
                 a_input_mask,
                 a_segment_ids,
                 a_pos_ids,
                 label):
        self.qas_id = qas_id
        self.q_tokens = q_tokens
        self.q_input_ids = q_input_ids
        self.q_input_mask = q_input_mask
        self.q_segment_ids = q_segment_ids
        self.q_pos_ids = q_pos_ids
        self.a_tokens = a_tokens
        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.a_pos_ids = a_pos_ids
        self.label = label


class Examples_To_Features_Converter(object):
    def __init__(self):
        pass

    def __call__(self,
                 examples,
                 tokenizer,
                 max_seq_length,
                 max_question_length,
                 max_answer_length):
        for(example_idx, example) in enumerate(examples):

            #tokenization_info = self.all_tokenization_info[example.id]

            question_tokens = tokenizer.tokenize(example.question_text)
            #assert question_tokens == tokenization_info['question_subtokens']

            if len(question_tokens) > max_question_length:
                question_tokens = question_tokens[0:max_question_length]

            answer_tokens = tokenizer.tokenize(example.answer_text)
            #assert answer_tokens == tokenization_info['answer_subtokens']

            if len(answer_tokens) > max_answer_length:
                answer_tokens = answer_tokens[0:max_answer_length]


            q_tokens = []
            q_segment_ids = []

            q_tokens.append("[CLS]")
            q_segment_ids.append(0)

            for token in  question_tokens:
                q_tokens.append(token)
                q_segment_ids.append(0)

            q_tokens.append("[SEP]")
            q_segment_ids.append(0)

            a_tokens = []
            a_segment_ids = []

            a_tokens.append("[CLS]")
            a_segment_ids.append(0)

            for token in answer_tokens:
                a_tokens.append(token)
                a_segment_ids.append(0)


            a_tokens.append("[SEP]")
            a_segment_ids.append(0)


            q_input_mask = [1] * len(q_tokens)
            a_input_mask = [1] * len(a_tokens)

            while len(q_tokens) < max_question_length+2:
                q_tokens.append("[PAD]")
                q_segment_ids.append(0)
                q_input_mask.append(0)

            while len(a_tokens) < max_answer_length+2:
                a_tokens.append("[PAD]")
                a_segment_ids.append(0)
                a_input_mask.append(0)

            q_input_ids = tokenizer.convert_tokens_to_ids(q_tokens)
            a_input_ids = tokenizer.convert_tokens_to_ids(a_tokens)


            feature = InputFeatures(
                qas_id=example.id,
                q_tokens=q_tokens,
                q_input_ids=q_input_ids,
                q_input_mask=q_input_mask,
                q_segment_ids=q_segment_ids,
                q_pos_ids=range(max_question_length+2),
                a_tokens=a_tokens,
                a_input_ids=a_input_ids,
                a_input_mask=a_input_mask,
                a_segment_ids=a_segment_ids,
                a_pos_ids=range(max_answer_length+2),
                label=example.label)

            yield feature

def read_qas_examples(input_file):
    """Read a qas json file into a list of qasExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    examples = []
    for entry in input_data:
      id = entry["id"]
      question_text = entry["question"]
      #question_entities_strset = set([entity_info["text"] for entity_info in entry["question_entities"]])
      answer_text = entry["answer"]
      #answer_entities_strset = set([entity_info["text"] for entity_info in entry["answer_entities"]])
      label = entry["label"]
      #print(type(question_text))
      example = QASExample(
          id = id,
          question_text=question_text,
          #question_entities_strset=question_entities_strset,
          answer_text=answer_text,
          #answer_entities_strset=answer_entities_strset,
          label = label)
      #print(type(example.question_text))
      examples.append(example)
    return examples

class DataProcessor(object):
    def __init__(self,
                 args,
                 max_seq_length,
                 max_question_length,
                 max_answer_length):
        self._bert_tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model)
        self.train_examples = None
        self.predict_examples = None
        self._max_seq_length = max_seq_length
        self._max_question_length = max_question_length
        self._max_answer_length = max_answer_length

    def get_examples(self, data_path):
        examples = read_qas_examples(input_file=data_path)
        return examples

    def get_features(self, examples):
        convert_examples_to_features = Examples_To_Features_Converter()
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self._bert_tokenizer,
            max_seq_length=self._max_seq_length,
            max_question_length=self._max_question_length,
            max_answer_length=self._max_answer_length)
        return features
    def data_generator(self, data_path, batch_size, phase):

        if phase == 'train':
            self.train_examples = self.get_examples(data_path)
        elif phase == 'predict':
            self.predict_examples = self.get_examples(data_path)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")
        
        def batch_reader(features, batch_size):
            batch,q_input_ids,q_token_type_ids,q_attention_mask,q_position_ids,a_input_ids,a_token_type_ids,a_attention_mask,a_position_ids,label = [],[],[],[],[],[],[],[],[],[]
            batch_data = {}
            for (index, feature) in enumerate(features):
                q_input_ids.append(feature.q_input_ids)
                q_token_type_ids.append(feature.q_segment_ids)
                q_attention_mask.append(feature.q_input_mask)
                q_position_ids.append(feature.q_pos_ids)
                a_input_ids.append(feature.a_input_ids)
                a_token_type_ids.append(feature.a_segment_ids)
                a_attention_mask.append(feature.a_input_mask)
                a_position_ids.append(feature.a_pos_ids)
                label.append(feature.label)

                to_append = len(q_input_ids) < batch_size

                if to_append:
                    continue
                else:
                    batch_data["q_input_ids"] = q_input_ids
                    batch_data["q_token_type_ids"] = q_token_type_ids
                    batch_data["q_attention_mask"] = q_attention_mask
                    batch_data["q_position_ids"] = q_position_ids
                    batch_data["a_input_ids"] = a_input_ids
                    batch_data["a_token_type_ids"] = a_token_type_ids
                    batch_data["a_attention_mask"] = a_attention_mask
                    batch_data["a_position_ids"] = a_position_ids
                    batch_data["label"] = label
                    batch.append(batch_data)
                    batch_data = {}
                    q_input_ids, q_token_type_ids, q_attention_mask, q_position_ids, a_input_ids, a_token_type_ids, a_attention_mask, a_position_ids, label = [], [], [], [], [], [], [], [], []
            return batch



        if phase == 'train':
            features = self.get_features(self.train_examples)
        else:
            features = self.get_features(self.predict_examples)
        
        return batch_reader(features,batch_size)
        