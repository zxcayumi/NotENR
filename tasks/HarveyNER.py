# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import re
import openai
import json
import string
import numpy as np
from collections import defaultdict
import random
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import Dataset
from vllm import LLM, SamplingParams
from passage_retrieval import Retriever
import sys
import re
from rouge import Rouge

sys.path.append("..")
# from run_uie import UIE_model

class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_fn(batch_data)

    def _collate_fn(self, batch_data):
        # This function will transform a batch of data into the desired format.
        question, answers = zip(*[(item['question'], item['answer']) for item in batch_data])  # Changed to tags
        return {'question': question, 'answer': answers, }

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
      
class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "tweebank",
                 task_description = "Find the named entity",
                 data_dir='',  
                 seed=42, 
                 query='',

                 post_instruction=True, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):
        self.options = {}
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        TaskDataset=TaskDataset,
                        option_num=option_num,
                        )




        # self.UIE_model, self.tokenizer = UIE_model()
        self.answer_format_prompt = ""        
        self.answer_format_prompt2 = "Output the score to one decimal place, for example, 4.5" 

    def get_dataloader(self, split, batch_size, shuffle=False):
        if split not in self.dataset:
            raise ValueError(f'Dataset split {split} does not exist.')

        dataset = self.dataset[split]
        return CustomDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def load_task_dataset(self, data_dir):
        # return load_dataset("ncbi_disease").filter(lambda example: len(example["ner_tags"]) > 0)
        return load_dataset(data_dir).filter(lambda example: example["label"] != 'None')

    def transform_format(self, data_dict):

        transformed_data_dict = {}
        
        for split_name, data_split in data_dict.items():
            print(f"Transforming {split_name} data...")
            
            formatted_data = []
            
            # Check available columns
            available_columns = data_split.column_names
            print(f"Available columns: {available_columns}")
            
            if 'sentence' in available_columns and 'label' in available_columns:
                '''
                RT @USER2362 : Farmall Heart Of The Holidays Tabletop Christmas Tree With Lights And Motion URL1087 #Holiday #Gifts
                [{'name': 'Farmall', 'pos': [15, 22], 'type': 'ORGANIZATION'}]
                '''
                for sentence, label in zip(data_split['sentence'], data_split['label']):
                    question = sentence
                    answer = label
                    formatted_example = {
                        'question': question,
                        'answer': answer
                    }
                    formatted_data.append(formatted_example)
                transformed_data_dict[split_name] = formatted_data
            else:
                print(f"Columns 'sentence' and/or 'entities' not found in {split_name} data_split.")

        
        return transformed_data_dict

    def build_forward_prompts_completion(self, questions, cur_propmt, type):
        if type == "rec":
            return super().build_forward_prompts_completion(questions, cur_propmt, self.answer_format_prompt)
        else:
            return super().build_forward_prompts_completion(questions, cur_propmt, self.answer_format_prompt2)


    def extract_entity(self, preds):
        batch_prompts = []
        for pred in preds:

            query = "I will give you a sentence. #Sentence: {0}\nPlease perform named entity recognition from the sentence.\nThe requirements are as follows:\n1. Let's think from following steps: First step, extract all candidate entities from the sentence. Second step, discard common nouns from candidate entities. Third step, classify remaining proper nouns in candidate entities. Option: exact location, area, road, river.\n2. Output candidate entities per step. format final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Echo Ln - Katy Fwy: exact location; Fort Bend: area;]\"\n".format(pred)

            batch_prompts.append(query)
            
        response = self.batch_forward_chatcompletion(batch_prompts)
        return response

    def cal_metric(self, preds, labels, questions):

        '''
            <task specific>
            return 1 number / tuple of metrics
        '''
        response = self.extract_entity(preds)

        rouge, NER_Prediction = self.cal_gen(response, labels)
        Micro_f1 = self.cal_f1(response, labels)

        return rouge, NER_Prediction, Micro_f1

    def cal_f1(self, preds, labels):
        if len(preds) != len(labels):
            print("Mismatched length between preds and labels")
            return []
        

        TP = 0
        FP = 0
        FN = 0

        total_num = 0 
        model_num = 0 

        for pred, label in zip(preds, labels):
            pred = self.change_format(pred)
            label = label.lower()

            pred_list = pred.split(';')
            labels_list = label.split('; ')
            total_num += label.count(';')
            model_num += pred.count(';')

            labels_dict = {}
            for item in labels_list:
                try:
                    key, value = item.rsplit(': ', maxsplit=1)
                    labels_dict[key] = value
                except:
                    continue
            
           
            for key in labels_dict:
                match_found = False
                for item in pred_list:
                    if key in item and labels_dict[key] in item:
                        TP += 1
                        match_found = True
                        break
                if not match_found:
                    FN += 1


        FP = model_num - TP
        print(TP, "xxx", FP, "xxx", FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        # precision = count / model_num
        # recall = count / total_num
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print(precision, recall, f1)
        # exit(0)
        return f1

    def gpt_chat_completion(self, **kwargs):
        backoff_time = 1
        while True:
            try:
                return openai.ChatCompletion.create(**kwargs)
            except openai.error.OpenAIError:
                print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
                time.sleep(backoff_time)
                backoff_time *= 1.5

    def batch_forward_chatcompletion(self, batch_prompts, model='gpt-3.5-turbo-16k', temperature=0, max_tokens=1024):
        responses = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt},]
            response = self.gpt_chat_completion(messages=messages, model=model, temperature=temperature)
            responses.append(response['choices'][0]['message']['content'].strip())
        return responses
    
    def cal_similarity(self, all_preds, all_questions):
        similarity = []
        eval_prompt = 'Score the semantic similarity between these two sentences from 0 to 5:'
        for pred, question in zip(all_preds, all_questions):
            
            text = ['1. ' + pred + ' 2. ' + question]
            batch_prompts = self.build_forward_prompts_completion(text, eval_prompt, "sim")
            # 怎么再用这个函数 用chatGPT去评价
            responses = self.batch_forward_chatcompletion(batch_prompts = batch_prompts)
            similarity.extend(responses)
        return similarity
    
    def remove_user_url_entries(self, input_string):
        
        pattern = r'\b(user\d+|url\d+):[^;]+;'

        result = re.sub(pattern, '', input_string)
        return result
    

    def change_format(self, _sentence):

        _sentence = _sentence.strip().lower()

        start_index = _sentence.rfind('[')
        end_index = _sentence.rfind(']')

        if start_index != -1 and end_index != -1:
            context = _sentence[start_index + 1:end_index]
        else:
            context = _sentence

        context = context.replace("\"","")

        context = context.split("; ")
        context = list(set(context))
        context = "; ".join(context)

        if context == '':
            context = 'None'
        if context[-1].isalpha():
            context = context + ';'
        else:
            context = context[:-1] + ';'
        return context.strip()
    

    def cal_correct(self, preds, labels):

        # Assuming both preds and labels are lists of lists
        if len(preds) != len(labels):
            print("Mismatched length between preds and labels")
            return []

        comparison_results = []
        
        for pred, label in zip(preds, labels):
            pred = self.change_format(pred)
            # Compare each pair of lists
            flag = 1
            label = label.lower()
            label = label.replace("; ", ";").strip()
            label_tuple = label.split(";")

            for _label in label_tuple:

                if _label not in pred:
                    flag = 0
                    break
            comparison_results.append(flag)

        return comparison_results
    
    def get_text_between_brackets(self, text):
        start_index = text.find(']')
        end_index = text.find('[', start_index)
        if start_index != -1 and end_index != -1:
            return text[start_index + 1: end_index]
        else:
            return None
        
    def cal_gen(self, preds, labels):
        reference = labels
        hypothesis = preds
        assert len(reference) == len(hypothesis)

        # rouge_list = []
        hyps = []
        refs = []

        for label, answer in zip(reference, hypothesis):
            label = label.strip().lower()
            
            answer = self.change_format(answer)

            hyps.append(answer)
            refs.append(label)

        rouge = Rouge()
        avg_rouge = rouge.get_scores(hyps, refs, avg=True)
            


        return avg_rouge, hyps
        
