import json
import openai
import time
import re

openai.api_base = "https://api.chatanywhere.com.cn"


def extract_strings_with_brackets(text):
    # 使用正则表达式匹配中括号中的内容
    pattern = r'\[(.*?)\]'
    matches = re.search(pattern, text)
    if(matches is None):
        return "None"
    else:
        return matches.group(1)

def openai_key_config(api_key=None):
    
    if api_key is not None:
        print(f'api_key: {api_key}')
        openai.api_key = api_key.strip()
        return
    
def batch_forward_chatcompletion(batch_prompts, model='gpt-3.5-turbo', temperature=0, max_tokens=1024):
    responses = []
    for prompt in batch_prompts:
        messages = [{"role": "user", "content": prompt},]
        response = gpt_chat_completion(messages=messages, model=model, temperature=temperature)
        responses.append(response['choices'][0]['message']['content'].strip())
    return responses

def gpt_chat_completion(**kwargs):
    backoff_time = 1
    while True:
        try:
            return openai.ChatCompletion.create(**kwargs)
        except openai.error.OpenAIError:
            print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5

def gpt_completion(**kwargs):
    backoff_time = 1
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5


def load_data(data_path):
    # 读取JSON文件
    with open(data_path, 'r') as file:
        data = json.load(file)

    # 初始化空列表
    sentences = []
    labels = []

    # 遍历数据
    for item in data:
        if item['label'] != 'None':
            sentences.append(item['sentence'])
            labels.append(item['label'])

    return sentences, labels

    # # 输出结果
    # print("Sentences:", sentences)
    # print("Labels:", labels)

def change_format(_sentence):

        _sentence = _sentence.strip().lower()
        
        _sentence = _sentence.replace(".;",";")
        _sentence = _sentence.replace(";.",";")
        _sentence = _sentence.replace(";;",";")
        start_index = _sentence.rfind('[')
        end_index = _sentence.rfind(']')

        # 提取中括号中的内容
        if start_index != -1 and end_index != -1:
            context = _sentence[start_index + 1:end_index]
        else:
            context = _sentence
        # pattern = r': (.*?);'
        # matches = list(re.finditer(pattern, _sentence))[::-1]
        
        # for match in matches:
        #     substring = match.group(1)
        #     if "misc" not in substring and "person" not in substring and "location" not in substring and "organization" not in substring:
        #         _sentence = _sentence[:match.start(1)] + "misc" + _sentence[match.end(1):]
        
        context = context.replace("miscellaneous", "misc")
        context = context.replace("person", "per")
        context = context.replace("location", "loc")
        context = context.replace("organization", "org")
        context = context.replace("\"","")

        context = context.split("; ")
        context = list(set(context))
        context = "; ".join(context)
        
        context = context.replace(".;",";")
        context = context.replace(";.",";")
        context = context.replace(";;",";")
        
        
        # answer = answer.replace(".;", ";").replace(".\n", "\n")
        # context = self.remove_user_url_entries(context)
        if context == '':
            context = 'None'
        if context[-1].isalpha():
            context = context + ';'
        else:
            context = context[:-1] + ';'
        return context.strip()


def cal_f1(preds, labels):
        
        if len(preds) != len(labels):
            print("Mismatched length between preds and labels")
            return []
        
        TP = 0
        FP = 0
        FN = 0
        # count = 0 # 模型正确识别的实体数量
        total_num = 0 # 数据中实际的实体总量
        model_num = 0 # 模型识别的实体总数
        #print(preds, "xx", labels)
        #exit(0)
        for pred, label in zip(preds, labels):
            # pred = change_format(pred)
            pred = pred.lower()
            label = label.lower()
            print(pred, '////////////' ,label)
            pred_list = pred[:-1].split('; ')
            labels_list = label[:-1].split('; ')
            total_num += label.count(';')
            model_num += pred.count(';')
            labels_dict = {item.rsplit(': ', maxsplit=1)[0]: item.rsplit(': ', maxsplit=1)[1] for item in labels_list}
           
            for key in labels_dict:
                match_found = False
                for item in pred_list:
                    if key in item and labels_dict[key] in item:
                        TP += 1
                        match_found = True
                        break
                if not match_found:
                    FN += 1
            # print(labels_dict, "xxxxx", pred_list)
            # exit(0)
            # for pred in pred_list:
            #     match_found = False
            #     for _label in labels_list:
            #         if pred == _label:
            #             TP += 1
            #             match_found = True
            #             break
            #     if not match_found:
            #         FN += 1

        FP = model_num - TP
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
        return precision, recall, f1

if __name__ == "__main__":

    openai_key_config("sk-U6trEjW9rRZHHQ6s2ArpGTRflrhzh7XNWQX9MY7XivbAIpZW")

    # data_path = '/home/hyy/codes/Text_Reconstruction/data/NER/For_RAG/Tweebank-NER-v1.0/test.json'
    data_path = '/home/hyy/codes/Text_Reconstruction/data/NER/For_RAG/WNUT17/test.json'
    sentences, labels = load_data(data_path)
    batch_prompts = []

    predictions = []

    for sentence in sentences:
        # prompt = "Please perform the named entity recognition to the following tweet. #Tweet: {0}\nformat final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Falling in love: miscellaneous; IMF: organization;]".format(sentence)
        #prompt = "Please list all entity words in the tweet that fit the category. Output format as a list \"[word1: type1; word2: type2;]\". Option: person, organization, location, miscellaneous; #Tweet: {0}".format(sentence)
        #query = "I will give you a sentence. #Sentence: {0}\nPlease perform named entity recognition from the sentence.\nOption: corporation, creative-work, group, location, person, product.\nOutput candidate entities per step. format final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Falling in love: creative-work; Willson: person;]\"\n".format(sentence)
        query = "I will give you a sentence. #Sentence: {0}\nPlease perform named entity recognition from the sentence.\nThe requirements are as follows:\n1. Let's think from following steps: First step, extract all candidate entities from the sentence. Second step, distinguish common nouns and proper nouns from candidate entities according to your knowledge. Third step, discard common nouns and retain proper nouns from candidate entities. Forth, classify remaining proper nouns in candidate entities. Option: corporation, creative-work, group, location, person, product.\n2. Output candidate entities per step. format final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Falling in love: creative-work; Willson: person;]".format(sentence)
        # prompt = "I will give you a sentence. #Sentence: {0}\nPlease perform named entity recognition from the sentence.\nThe requirements are as follows:\n1. Let's think from following steps: First step, extract all candidate entities from the sentence. Second step, distinguish common nouns and proper nouns from candidate entities according to your knowledge. Third step, discard common nouns and retain proper nouns from candidate entities. Forth, classify remaining proper nouns in candidate entities. Option: person, organization, location.\n2. Output candidate entities per step. Format final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Falling in love: miscellaneous; IMF: organization;]\"\n".format(sentence)
        batch_prompts.append(query)
    print(len(sentences), len(labels))
    response = batch_forward_chatcompletion(batch_prompts)

    for answer in response:
        predictions.append(extract_strings_with_brackets(answer))
    # print(len(labels), len(predictions))
    precision, recall, f1 = cal_f1(predictions, labels)
    print(precision, recall, f1)
    # print(extract_strings_with_brackets(response[0]))
    # exit(0)
    
