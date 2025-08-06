# log prompts
forward_log_tempelate = """---------------\tforward\t----------------
cur_prompt:\n{cur_prompt}
batch_prompts:\n{batch_prompts}
responses:\n{responses}
labels:  {labels}
NMI: {NMI}
ARI: {ARI}
AMI: {AMI}
ACC: {ACC}
"""
forward_log_tempelate_v1 = """---------------\tforward\t----------------
cur_prompt:\n{cur_prompt}
labels:  {labels}
preds:   {preds}
rouge1_F1: {rouge1_F1}
rougeL_F1:     {rougeL_F1}
"""

error_string_log_tempelate = """--------------\terror_string\t-----------------
{error_string}
"""

correct_string_log_tempelate = """--------------\tcorrect_string\t-----------------
{correct_string}
"""

gradient_log_tempelate = """---------------\tcal_gradient\t----------------
gradient_prompt:\n{gradient_prompt}
gradient:\n{gradient}
"""

correct_gradient_log_tempelate = """---------------\tcal_correct_gradient\t----------------
gradient_prompt:\n{gradient_prompt}
gradient:\n{gradient}
"""

optimize_log_tempelate = """-------------\toptimize\t---------------
optimize_prompt:\n{optimize_prompt}
response:\n{response}
optimized_prompt:\n{optimized_prompt}
"""

self_evaluate_log_tempelate = """-------------\tself_eval\t---------------
self_eval_prompt:\n{self_eval_prompt}
response:\n{response}
self_eval_score:\n{self_eval_score}
"""
