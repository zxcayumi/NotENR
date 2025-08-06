from .gradient_descent import *
from typing import Generic
from ..search_algo.base_algo import State, Action
from ..search_algo.mcts import MCTSNode
from tqdm import tqdm
class WorldModel(Generic[State, Action]):
    def __init__(self,
                 task,
                 logger,
                 
                 # model
                 pred_model: str,
                 optim_model: str,
                 pred_temperature: float, 
                 optim_temperature: float,
                 
                 max_tokens=2048,
                 num_new_prompts = 2,
                 
                 train_shuffle = True,
                 train_batch_size: int = 5,
                 test_batch_size: int = 2,
                 eval_batch_size: int = 1,
                 **kwargs) -> None:
        
        self.task = task
        self.logger = logger
        self.pred_model = pred_model
        self.pred_temperature=pred_temperature
        self.optim_model = optim_model
        self.optim_temperature = optim_temperature
        self.max_tokens=max_tokens
        self.num_new_prompts = num_new_prompts
        
        self.train_shuffle = train_shuffle
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.eval_batch_size = eval_batch_size
        
        self.train_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=train_batch_size, 
                                                        shuffle=train_shuffle)
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        
        self.test_dataloader = self.task.get_dataloader('test', 
                                                        batch_size=test_batch_size, 
                                                        shuffle=False)
        self.eval_dataloader = self.task.get_dataloader('eval', 
                                                        batch_size=eval_batch_size, 
                                                        shuffle=False)
        self.gradient_descent = GradientDescent(task=self.task, 
                                                logger=self.logger, 
                                                pred_model=pred_model, 
                                                optim_model=optim_model, 
                                                num_new_prompts = num_new_prompts,
                                                forward_temperature=pred_temperature, 
                                                optim_temperature = optim_temperature,
                                                max_tokens=max_tokens,
                                                )
        self.log_vars()
        
    def log_vars(self):
        self.logger.info('----------------- World Model --------------------------')
        ignored_print_vars = ['task', 'logger', 'train_dataloader','train_data_iterator','test_dataloader','eval_dataloader', 'gradient_descent']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')
        
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
                
    def get_train_batch(self):
        return next(self.train_data_iterator)
    
    # 从当前node一路往父节点收集prompt，prompt反向(从根节点到当前node)返回。
    def _get_trajectory_prompts(self, node: MCTSNode):
        trajectory_prompts = []
        temp_node = node
        while True:
            trajectory_prompts.append(temp_node.prompt)
            if temp_node.parent is not None:
                temp_node = temp_node.parent
            else:
                break
        return trajectory_prompts[::-1]
                
    
    def _gradient_descent_step(self, node: MCTSNode, batch):
        trajectory_prompts = self._get_trajectory_prompts(node=node)
        # helper_data形式{trajectory_prompts: [...]}
        helper_data = dict(trajectory_prompts=trajectory_prompts)
        
        gradient_descent_output = self.gradient_descent(batch, node.prompt, helper_data) 
        if gradient_descent_output['acc']==-1:
            return [], gradient_descent_output
        
        new_nodes = []
        for prompt in gradient_descent_output['optimized_prompts']:
            child_node = MCTSNode(
                prompt=prompt, 
                action=gradient_descent_output['gradient'], 
                parent=node)
            new_nodes.append(child_node)
        
        return new_nodes, gradient_descent_output
    
    def step(self, node:MCTSNode, batch):
        new_nodes, gradient_descent_output = self._gradient_descent_step(node=node, batch=batch)
        return new_nodes, gradient_descent_output
    
    def build_root(self, init_prompt):
        node = MCTSNode(prompt=init_prompt, action=None, parent=None)
        node.reward = self._reward_type_helper(self.evaluate_prompt(prompt=node.prompt)["metric"])
        return node

    def evaluate_child_node(self, node:MCTSNode):
        evaludate_output = self.evaluate_prompt(prompt=node.prompt)
        node.reward = self._reward_type_helper(evaludate_output["metric"])
    
    def test_prompt(self, prompt):
        metric, eval_output = self.eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           model=self.pred_model,
                                           temperature=self.pred_temperature,
                                           max_tokens=self.max_tokens)
        return metric, eval_output
    
    def _reward_type_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
        

    def evaluate_prompt(self, prompt):
        self.logger.info(f'prompt: {prompt}')
        # 在这里计算指标
        metric, eval_output = self.eval_instruction_with_loader(
            task=self.task, 
            eval_prompt=prompt,
            dataloader=self.eval_dataloader,
            model=self.pred_model,
            temperature=self.pred_temperature,
            max_tokens=self.max_tokens
            )
        correct = eval_output['correct']
        evaludate_output = dict(
            metric = metric,
            correct = correct,
            acc = np.mean(correct)
        )
        
        return evaludate_output
    
    def eval_instruction_with_loader(self, task, eval_prompt, dataloader, model='gpt-3.5-turbo', temperature=0, max_tokens=1024, record_outputs=True):

        build_forward_prompts_func = task.build_forward_prompts_completion
        if model in COMPLETION_MODELS:
            batch_forward_func = batch_forward_completion
        elif model in CHAT_COMPLETION_MODELS:
            batch_forward_func = batch_forward_chatcompletion
        elif model in OPENSOURCE_MODELS:
            batch_forward_func  = batch_forward_flant5
        else:
            raise ValueError(f"Model {model} not supported.")
        
        all_questions = []
        all_labels = []
        all_preds = []
        all_prompts = []
        all_responses = []
        eval_output = {}

        
        pbar = tqdm(dataloader, leave=False)
        for batch in pbar:
            '''
            batch:{'question': (['A', 'G', '-', 'to', '-', 'C', 'transition', 'at', 'the', 'second', 'base', 'of', 'codon', '413', 'in', 'exon', '12', 'of', 'the', 'gene', 'results', 'in', 'the', 'substitution', 'of', 'Pro413', 'for', 'Arg413', 'in', 'the', 'mutant', 'protein', '.'],), 
                   'answer': ([],)}
            '''

            ''' chatGPT重构+InstructUIE抽取 '''

            batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt, task.answer_format_prompt)
            '''
            batch_prompts:["Correct the non-standard parts of the following sentence.\n
            @USER721 I remember when you had like a thousand. You have the juice now.\n
            Output the standardized sentence."]

            You have the juice now.  =  你现在很厉害、很有能力了。
            '''
            responses = batch_forward_func(batch_prompts, model=model, temperature=temperature, max_tokens=max_tokens)
            # CHATGPT回答结果
            # ['@USER721 I remember when you had like a thousand. You have the juice now.']
            ''' preds应该是重构好的文本,例如['sentence1','sentence2'] '''
            
            preds = responses
            labels = batch['answer'][0]

            # 这里返回的preds应该是重构好的文本了
            all_preds.extend(preds)
            # NER的ground_truth
            all_labels.append(labels)
            # NER的原句
            all_questions.extend(batch['question'])


            if record_outputs:
                all_prompts.extend(batch_prompts)
                all_responses.extend(responses)

            metric = task.cal_metric(all_preds, all_labels, all_questions)
            
            if not isinstance(metric, tuple):
                pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
            else:
                pbar.set_postfix_str(f"Test Metrics: {metric}")
        
        if record_outputs:
            eval_output['model_inputs'] =  all_prompts
            eval_output['model_responses'] =  all_responses
            eval_output['preds'] =  all_preds
            eval_output['labels'] =  all_labels

        eval_output['correct'] =  task.cal_correct(all_preds, all_labels)    
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        return metric, eval_output