from .prompts import *
from .prompts.world_model_prompts import *
from ..utils import *
import re
import numpy as np


class GradientDescent():
    def __init__(self,
                 task,

                 pred_model,
                 optim_model,
                 forward_temperature=0,
                 optim_temperature=0,
                 max_tokens=2048,
                 print_log=True,
                 logger=None,
                 num_new_prompts=1, ):

        self.task = task
        self.pred_model = pred_model
        self.optim_model = optim_model
        self.forward_temperature = forward_temperature
        self.optim_temperature = optim_temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.num_new_prompts = num_new_prompts

        self.use_correct_examples = False

        self.optimize_prompt_tempelate = optimize_prompt_tempelate_single if num_new_prompts == 1 else optimize_prompt_tempelate
        self.gradient_prompt_tempelate = gradient_prompt_tempelate
        self.error_example_template = error_example_template

        self.example_temlate = example_template_v0

        self._build_forward_prompts_func = task.build_forward_prompts_completion

        # CHAT_COMPLETION_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314']
        # COMPLETION_MODELS =  ['text-davinci-003', 'text-davinci-002','code-davinci-002']
        # OPENSOURCE_MODELS = ['flan-t5']
        # PALM_MODELS = ['models/chat-bison-001']

        # 给模型输入一个batch，模型返回一个batch
        if pred_model in COMPLETION_MODELS:
            self._batch_forward_func = batch_forward_completion
        elif pred_model in CHAT_COMPLETION_MODELS:
            self._batch_forward_func = batch_forward_chatcompletion
        elif pred_model in OPENSOURCE_MODELS:
            self._batch_forward_func = batch_forward_flant5
        else:
            raise ValueError(f"Model {pred_model} not supported.")

    # 返回ChatGPT生成的结果
    def forward(self, batch, cur_prompt):

        batch_size = len(batch['question'])
        # 为句子生成prompt
        batch_prompts = self._build_forward_prompts_func(
            batch['question'], cur_prompt, "rec")
        # 获得chatGPT生成的结果
        responses = self._batch_forward_func(batch_prompts, model=self.pred_model, temperature=self.forward_temperature,
                                             max_tokens=self.max_tokens)

        labels = batch['answer']
        # preds是要进行一下抽取的
        # intent_Prediction = self.task.intent_discovery(preds)

        metrics, new_responses, _ = self.task.cal_metric(responses, labels)

        correct = self.task.cal_correct(new_responses, labels)
        # print(correct)
        # exit(0)
        NMI = metrics[0]
        ARI = metrics[1]
        ACC = metrics[2]
        AMI = metrics[3]

        if np.mean(correct) == 1:
            return dict(acc=-1)

        batch_logs = []
        for i in range(batch_size):
            batch_logs.append({
                'cur_prompt': cur_prompt,
                'question': batch['question'][i],
                'model_input': batch_prompts[i],
                'gt_answer': batch['answer'][i],
                'model_response': new_responses[i],
                'label': labels[i],
                'refactor_text': responses[i]
            })

        forward_output = {
            'cur_prompt': cur_prompt,
            'acc_cluster': ACC,
            'nmi': NMI,
            'ari': ARI,
            'ami': AMI,
            'acc': np.mean(correct),
            'correct': correct,
            'examples': batch_logs,
        }

        if self.print_log:
            log_str = forward_log_tempelate.format(cur_prompt=cur_prompt,
                                                   batch_prompts=batch_prompts,
                                                   responses=new_responses,
                                                   labels=labels,
                                                   ACC=forward_output['acc_cluster'],
                                                   NMI=forward_output['nmi'],
                                                   ARI=forward_output['ari'],
                                                   AMI=forward_output['ami']
                                                   )

            self.logger.info(log_str)
        return forward_output

    def _get_batch_examples_str(self, batch):
        batch_example_strings = []
        for i in range(len(batch['question'])):
            batch_example_strings.append(self.example_temlate.format(index=i + 1,
                                                                     question=batch['question'][i],
                                                                     label=batch['answer'][i]))
        return ''.join(batch_example_strings)

    def _clean_self_eval_score(self, response):
        return re.findall(r'\d+', response)[-1]

    def _get_error_examples(self, forward_output):
        error_examples = []
        count = 0
        for i, example in enumerate(forward_output['examples']):
            if forward_output['correct'][i] == 0:
                count += 1
                print(example)
                error_examples.append(self.error_example_template.format(
                    index=count,
                    question=example['model_input'],
                    label=example['label'].lower(),
                    response=example['refactor_text'],
                    prediction=example['model_response']))
            elif forward_output['correct'][i] == 1:
                continue
            else:
                raise ValueError(
                    f'_get_error_examples: invalid correct number {i} {forward_output}.')
        error_string = ''.join(error_examples)
        return error_string

    def _optim_model_completion(self, model_input):
        messages = [{"role": "user", "content": model_input}, ]
        response = gpt_chat_completion(messages=messages,
                                       model=self.optim_model,
                                       temperature=self.optim_temperature).choices[0].message.content.strip()
        return response

    def _build_prompt_trajectory_str(self, prompts):
        prompt_path_str = ""
        prompt_path_str_tempelate = "({index}) {prompt}\n"
        for i, prompt in enumerate(prompts):
            prompt_path_str += prompt_path_str_tempelate.format(
                index=i, prompt=prompt)
        return prompt_path_str

    def cal_gradient(self, cur_prompt, error_string):
        correct_gradient = None
        '''
        gradient_prompt_tempelate = """
        I'm writing prompts for a language model designed for a task.

        My current prompt is:
        {cur_prompt}

        But this prompt gets the following examples wrong:
        {error_string}

        For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
        """.strip()

        '''
        gradient_prompt = self.gradient_prompt_tempelate.format(cur_prompt=cur_prompt,
                                                                error_string=error_string)
        # 针对错误,生成Error feedback
        gradient = self._optim_model_completion(gradient_prompt)

        if self.print_log:
            log_str = gradient_log_tempelate.format(gradient_prompt=gradient_prompt,
                                                    gradient=gradient)

            self.logger.info(log_str)

        return gradient, correct_gradient

    def _clean_optim_response(self, optim_response):
        pattern = r'<START>(.*?)<END>'
        matches = re.findall(
            pattern=pattern, string=optim_response, flags=re.DOTALL)
        for i, m in enumerate(matches):
            matches[i] = m.strip()
        return matches

    def optimize(self, cur_prompt, error_str, gradient, trajectory_prompts, correct_gradient, steps_per_gradient):
        optimize_prompt = self.optimize_prompt_tempelate.format(cur_prompt=cur_prompt,
                                                                correct_gradient=correct_gradient,
                                                                error_str=error_str,
                                                                gradient=gradient,
                                                                trajectory_prompts=trajectory_prompts,
                                                                steps_per_gradient=steps_per_gradient)
        response = self._optim_model_completion(optimize_prompt)
        optimized_prompt = self._clean_optim_response(response)
        if self.print_log:
            log_str = optimize_log_tempelate.format(optimize_prompt=optimize_prompt,
                                                    response=response,
                                                    optimized_prompt=optimized_prompt)
            self.logger.info(log_str)

        return optimized_prompt

    def gradient_descent_step(self, cur_prompt, batch, helper_data):

        self.logger.info(f'cur_prompt: {cur_prompt}')

        # gradient_descent_output = {
        #     'cur_prompt': cur_prompt,
        #     'correct':correct,
        #     'examples':batch_logs,
        #     'acc':np.mean(correct)
        #     }

        # 返回chatGPT生成的结果
        gradient_descent_output = self.forward(
            batch=batch, cur_prompt=cur_prompt)
        print("*"*50)
        print("1. gradient_descent_output")
        print(gradient_descent_output)
        if gradient_descent_output['acc'] == -1:
            return gradient_descent_output

        # 找到错误的样本,分类错误
        error_string = self._get_error_examples(gradient_descent_output)
        print("*" * 50)
        print("2. error_string")
        print(error_string)
        # 根据错误样本生成Error feedback
        gradient, correct_gradient = self.cal_gradient(
            cur_prompt=cur_prompt, error_string=error_string)

        print("*" * 50)
        print("3. gradient")
        print(gradient)

        print("*" * 50)
        print("4. correct_gradient")
        print(correct_gradient)

        trajectory_prompts = helper_data['trajectory_prompts']
        trajectory_prompts = self._build_prompt_trajectory_str(
            trajectory_prompts)

        # 根据Error feedback,优化出新prompt
        optimized_prompts = self.optimize(cur_prompt=cur_prompt,
                                          error_str=error_string,
                                          gradient=gradient,
                                          trajectory_prompts=trajectory_prompts,
                                          correct_gradient=correct_gradient,
                                          steps_per_gradient=self.num_new_prompts)
        print("*" * 50)
        print("5. optimized_prompts")
        print(optimized_prompts)
        gradient_descent_output['error_string'] = error_string
        gradient_descent_output['gradient'] = gradient
        gradient_descent_output['optimized_prompts'] = optimized_prompts
        return gradient_descent_output

    # 入口
    def __call__(self, batch, cur_prompt, helper_data=None):
        gradient_descent_output = self.gradient_descent_step(cur_prompt=cur_prompt, batch=batch,
                                                             helper_data=helper_data)
        return gradient_descent_output
