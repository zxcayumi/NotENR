import argparse
from prompt_optim_agent import *

openai.api_base = "https://api.chatanywhere.com.cn"

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description='Process prompt search agent arguments')
    parser.add_argument('--task', type=str, default='ner',  help='The task you want to do.')  
    # 任务名称
    parser.add_argument('--task_name', type=str, default='bigbench',  help='This is consistent to the task file names in tasks folder. The default is bigbench task.')  
    # 搜索策略，蒙特卡洛搜索或束搜索——可改用AlphaGo 算法或遗传算法等
    parser.add_argument('--search_algo', type=str, default='mcts', choices=['mcts', 'beam_search'], help='Prompt search algorithm. Choose from \'mcts\' and \'beam_search\'.')    
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size depending on the memory and model size')
    # 最深搜索路径
    parser.add_argument('--depth_limit', type=int, default=5, help="The max depth of a single searching path.")
    parser.add_argument('--train_size', type=int, default=150, help="The dataset that sample batches from.")
    parser.add_argument('--eval_size', type=int, default=100, help="Calculate reward on this set.")
    parser.add_argument('--test_size', type=int, default=None, help="Test set size.")
    parser.add_argument('--seed', type=int, default=42, help="The seed to shuffle the dataset. Set it for a fixed test set.")
    parser.add_argument('--train_shuffle', type=str2bool, default=True, help='Shuffle training set')
    
    #Search
    # 初始prompt，根节点
    parser.add_argument('--init_prompt', type=str, default="Let's answer the question.", help='Initial prompt written by human.')
    parser.add_argument('--iteration_num', type=int, default=12, help='MCTS iteration number.')
    # 每次扩展batch的采样数
    parser.add_argument('--expand_width', type=int, default=3, help="The number of batches sampled in each expansion.")
    # 一次错误反馈产生的新prompt个数
    parser.add_argument('--num_new_prompts', type=int, default=1, help="The number of new prompts sampled in each batch.")
    # 单个prompt长度
    parser.add_argument('--prompt_length_limit', type=int, default=300)
    parser.add_argument('--post_instruction', type=str2bool, default=False, help="True: the position of instruction is question+instruction; \nFalse: the position of instruction is instruction+question")
    # MCTS
    parser.add_argument('--min_depth', type=int, default=3, help="Early stop depth: early stop is only applied when depth is larger than min_depth.")
    parser.add_argument('--w_exp', type=float, default=2.5, help="Weight of MCTS.")
    
    # BeamSearch
    # 束搜索宽度k
    parser.add_argument('--beam_width', type=int, default=3)

    # World Model
    # 所用大语言模型，用来做预测(实体抽取等)
    parser.add_argument('--pred_model', type=str, default='gpt-3.5-turbo', help='The base model that makes predictions.')
    # 所用大语言模型，用来做生成更好的prompt
    parser.add_argument('--optim_model', type=str, default='gpt-4o', help='Prompt optimizer.') 
    parser.add_argument('--pred_temperature', type=float, default=0.0)
    parser.add_argument('--optim_temperature', type=float, default=1.0)
    
    parser.add_argument('--log_dir', type=str, default='../logs/', help='Log directory.')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data file (if needed)')
    parser.add_argument('--api_key', type=str, default=None, help='openai api key')
    parser.add_argument('--openai_key_txt_file', type=str, default='../api_keys.txt', help='The txt file that contains openai api key.')
    
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    args = vars(args)
    
    openai_key_config(args['api_key'], args['openai_key_txt_file'])
    
    return args

# def create_prompt(task):
#     task_list = ['NER', 'EE']
#     assert task in task_list
#     if task == 'NER':
#         init_prompt = 'Correct the non-standard parts of the following sentence.'
#     else:
#         init_prompt = 'Correct the non-standard parts of the following sentence.'


def main(args):
    agent = BaseAgent(**args)
    agent.run(init_state=args['init_prompt'], iteration_num=args['iteration_num'])
    return

if __name__ == '__main__':

    args = config()
    main(args)
