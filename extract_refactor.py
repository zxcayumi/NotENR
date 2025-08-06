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

        for pred, label in zip(preds, labels):
            pred = change_format(pred)
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
    data_path = '/home/hyy/codes/Text_Reconstruction/data/NER/For_RAG/BTC/test.json'
    sentences, labels = load_data(data_path)
    batch_prompts = []

    predictions = []


    for sentence in sentences:
        # prompt = "Please perform the named entity recognition to the following tweet. #Tweet: {0}\nformat final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Falling in love: miscellaneous; IMF: organization;]".format(sentence)
        # prompt1 = '''It is your task to analyze the given tweet, focusing primarily on facilitating named entity recognition. This involves identifying, separating, and classifying entities, such as locations, individuals, organizations, etc., embedded within the sentence. The aim is to reformat the sentence in a way that distinctly illuminates these main entities while maintaining its original connotation. In cases where entities are part of larger phrases or multi-word entities, ensure you recognize and indicate as such. Particularly, be mindful of genitive constructs, such as "'s", where entities like a person and an organization might be linked and should be considered separately. Also, be aware of entities that might be attached to other words, for instance in the case of "Moroccan" in "Moroccan restaurant". Separate these and recognize them individually. This task isn't restricted to just rewriting or excluding elements like URLs and usernames from the tweet, but is majorly about appreciating the entities, understanding their context, and enhancing their prominence in the sentence structure. Always remember to distinguish between common nouns and named entities; the aim is to recognize the named entities specifically. Please return only the restructured sentence that ensures all key entities are individually identifiable and their original context is retained. #Tweet: {0}'''.format(sentence)
        prompt1 = '''Your task is to identify and categorise all entities in the tweet, remove URLs and usernames, and then rewrite the tweet without internet slang and abbreviations. The rewritten tweet should adhere to standard English grammar rules and be clear and easy to understand. Pay close attention to symbols and signs such as '@' and '#' as they may represent important entities. Finally, rewrite the tweet with suitable modifications for efficient named entity recognition, and then present the rewritten sentence. #Tweet: {0}'''.format(sentence)
        # print(prompt1)
        # exit(0)
        batch_prompts.append(prompt1)

    
    response = batch_forward_chatcompletion(batch_prompts)
    print(len(response), len(labels))
    print(response)
    # print(prompt1)
    # exit(0)
    batch_prompts = []
    for sentence in response:

        # prompt = "Please perform the named entity recognition to the following tweet. #Tweet: {0}\nformat final answer as a list like \"[word1: category1; word2: category2;]\". An example output is: \"[Falling in love: miscellaneous; IMF: organization;]".format(sentence)
        prompt2 = "Please list all entity words in the tweet that fit the category. Output format as a list \"[word1: type1; word2: type2;]\". Option: person, organization, location, miscellaneous. #Tweet: {0}".format(sentence)
        
        batch_prompts.append(prompt2)
    
    response = batch_forward_chatcompletion(batch_prompts)
    print("\n\n")
    print(response)
    print("\n\n")
    # print(len(response), len(labels))
    for answer in response:
        predictions.append(extract_strings_with_brackets(answer))
    # print(len(labels), len(predictions))
    # print(predictions)
    precision, recall, f1 = cal_f1(predictions, labels)
    print(precision, recall, f1)
        # print(extract_strings_with_brackets(response[0]))
        # exit(0)
    