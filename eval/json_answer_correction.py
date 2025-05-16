import json
import re
import argparse
import csv
import time
from collections import defaultdict
from openai import AzureOpenAI
from tqdm import tqdm
import os

class SimpleGenerator:
    def __init__(self, api_key, api_version, endpoint, engine):
        self.client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        self.engine = engine

    def generate_answer(self, question, max_retries=5, retry_delay=5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    temperature=0,
                    messages=[{"role": "user", "content": question}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "RateLimitError" in str(type(e)) and attempt < max_retries - 1:
                    try:
                        retry_after = int(str(e).split("Please retry after ")[1].split(" milliseconds")[0]) / 1000
                        time.sleep(retry_after)
                    except:
                        time.sleep(retry_delay)
                else:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(retry_delay)

# 提取答案和位置的函数（带位置限制）
def extract_answer_with_position(output, max_distance=50):
    """提取答案及其在文本中的位置，带结尾距离限制"""
    pattern = r'\[\[([A-F])\]\]'
    match = re.search(pattern, output)
    if match:
        answer = match.group(1)
        position = match.start(1)
        return answer, position
    
    # 提取所有大写字母及其位置
    uppercase_matches = list(re.finditer(r'[A-F]', output))
    if uppercase_matches:
        last_match = uppercase_matches[-1]
        answer = last_match.group(0)
        position = last_match.start()
        # 检查最后一个大写字母是否在结尾50字符内
        if len(output) - position - 1 <= max_distance:
            return answer, position
        else:
            return "", -1  # 位置超出限制
    
    return "", -1  # 无匹配

# GPT评测相关配置
API_KEY = ""
ENDPOINT = 'https://xxx.com/'
ENGINE = 'gpt-xxx'
api_version = "xxxxxxxx"
generator = SimpleGenerator(API_KEY, api_version, ENDPOINT, ENGINE)

# 选择题评测函数
def evaluate_choice_question(question, options, correct_answer, model_output):
    """使用GPT评估选择题答案"""
    prompt = f"""
你是一个评测助手。请判断下面模型输出的答案是否正确。

题目：
{question}

选项：
"""
    for k, v in options.items():
        prompt += f"{k}. {v}\n"

    prompt += f"""
正确答案是：{correct_answer}，该选项对应的内容是：{options[correct_answer]}

模型输出内容：
{model_output}

请你从模型输出中提取它的最终答案，并判断是否与正确答案一致。注意！模型会对该题进行详细地分析，你只需要评测最终答案是否正确即可！
如果正确，回复"1"，否则回复"0"。
只回复数字，不要多余内容。
"""

    try:
        ans = generator.generate_answer(prompt)
        return 1 if ans == "1" else (0 if ans == "0" else 2)
    except Exception as e:
        print(f"GPT调用错误: {e}")
        return 2

# 开放题评测函数
def evaluate_open_question(question, correct_option_content, model_output):
    """使用GPT评估开放题答案"""
    prompt = f"""
你是一个评测助手。请判断下面模型输出的答案是否正确。

题目：
{question}

正确答案内容：
{correct_option_content}

模型输出内容：
{model_output}

请你从模型输出中提取它的最终答案，并判断是否与正确答案内容一致。注意！意思大概一致即可算作正确。不用严格完全一致，需要仔细考虑模型的回答是否合理正确。
如果回答正确，回复"1"，否则回复"0"。
只回复数字，不要多余内容。
"""

    try:
        ans = generator.generate_answer(prompt)
        return 1 if ans == "1" else (0 if ans == "0" else 2)
    except Exception as e:
        print(f"调用模型接口出错: {e}")
        return 2

# 从JSONL文件加载开放题ID列表
def load_open_question_ids(jsonl_path):
    """从JSONL文件加载开放题ID列表"""
    open_question_ids = set()
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    open_question_ids.add(item['id'])
        print(f"已从{jsonl_path}加载{len(open_question_ids)}个开放题ID")
        return open_question_ids
    except Exception as e:
        print(f"加载开放题ID文件出错: {e}")
        return set()

# 更新答案正确性的主函数
def update_answer_correctness(json_data, is_open_question=False, use_gpt=False, max_distance=50, open_question_ids=None):
    """更新 JSON 数据中的评测结果并统计分类准确率"""
    total_questions = defaultdict(int)  # 按类别统计总题数
    correct_questions = defaultdict(lambda: defaultdict(int))  # 按模型和类别统计正确题数
    total_correct = defaultdict(int)  # 按模型统计总正确题数
    grand_total = 0  # 总题数
    
    # 定义要统计的类别
    categories = [
        "Geography", "Art", "Culture", "Social science", 
        "History", "Economy"
    ]
    
    # 跟踪处理的题目数量
    processed_count = 0
    # 存储已评测的题目
    evaluated_items = []
    
    for item in tqdm(json_data, desc="处理进度"):
        try:
            # 检查是否为需要处理的开放题
            if is_open_question and open_question_ids is not None:
                item_id = item.get('id')
                if item_id not in open_question_ids:
                    continue  # 不是指定的开放题，跳过
            
            # 获取题目类别
            category = item.get("category", "Other")
            
            results = item.get('results', {})
            if not results: 
                continue
                
            model_name = next(iter(results.keys()))
            model_output = results[model_name].get("output", "")
            correct_answer = item.get("correct_answer", "").strip().upper()
            
            if not correct_answer:
                print(f"警告：题目ID{item.get('id','未知')} 正确答案为空，跳过")
                continue
            
            # 只有使用GPT评估的题目才会被保留
            if use_gpt:
                if is_open_question:
                    # 开放题处理逻辑
                    correct_answer_list = [ans.strip() for ans in correct_answer.split(",") if ans.strip()]
                    if not correct_answer_list:
                        print(f"警告：题目ID{item.get('id','未知')} 正确答案为空，跳过")
                        continue
                    
                    correct_option_contents = []
                    options = item.get("options", {})
                    for ans_letter in correct_answer_list:
                        option_content = options.get(ans_letter, "").strip()
                        if option_content:
                            correct_option_contents.append(option_content)
                        else:
                            print(f"警告：题目ID{item.get('id','未知')} 正确答案选项'{ans_letter}'不存在或为空，跳过")
                            correct_option_contents = []
                            break
                    if not correct_option_contents:
                        continue
                        
                    correct_option_content = ",".join(correct_option_contents)
                    
                    # 调用GPT评估开放题
                    eval_result = evaluate_open_question(
                        item.get("question", ""),
                        correct_option_content,
                        model_output
                    )
                else:
                    # 选择题处理逻辑
                    eval_result = evaluate_choice_question(
                        item.get("question", ""),
                        item.get("options", {}),
                        correct_answer,
                        model_output
                    )
                
                is_correct = 1 if eval_result == 1 else 0
                # 更新GPT评测结果
                results[model_name]['gpt-4.1_eval'] = eval_result
                
                # 只保留使用GPT评估的题目
                evaluated_items.append(item)
            else:
                # 正则表达式评测模式
                answer, position = extract_answer_with_position(model_output, max_distance)
                is_correct = 1 if answer == correct_answer else 0
                # 更新正则匹配结果
                results[model_name]['extracted_answer'] = answer
            
            # 在正则模式下刷新is_correct属性
            results[model_name]['is_correct'] = bool(is_correct)
            
            # 统计分类准确率
            total_questions[category] += 1
            grand_total += 1
            processed_count += 1
            
            if is_correct:
                correct_questions[model_name][category] += 1
                total_correct[model_name] += 1
                
        except Exception as e:
            print(f"处理错误：{str(e)}")
    
    print(f"处理完成：共处理{processed_count}个题目，其中{grand_total - processed_count}个题目被过滤")
    
    # 如果使用GPT评估，返回已评估的题目列表
    if use_gpt:
        return {
            'total_questions': total_questions,
            'correct_questions': correct_questions,
            'total_correct': total_correct,
            'grand_total': grand_total,
            'categories': categories,
            'evaluated_items': evaluated_items
        }
    else:
        return {
            'total_questions': total_questions,
            'correct_questions': correct_questions,
            'total_correct': total_correct,
            'grand_total': grand_total,
            'categories': categories
        }

def save_accuracy_to_csv(all_stats, csv_path):
    """将多个模型的准确率统计保存到CSV文件"""
    # 合并所有类别的统计
    all_categories = set()
    all_model_names = set()
    
    for stats in all_stats:
        all_categories.update(stats['categories'] + ['Other', 'Overall'])
        all_model_names.update(stats['correct_questions'].keys())
    
    all_categories = sorted(all_categories)
    all_model_names = sorted(all_model_names)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        header = ['Category']
        for model_name in all_model_names:
            header.append(f'{model_name} Accuracy')
            header.append(f'{model_name} Correct')
            header.append(f'{model_name} Total')
        writer.writerow(header)
        
        # 写入每个类别的统计
        for category in all_categories:
            row = [category]
            
            for model_name in all_model_names:
                # 查找该模型的统计数据
                model_stats = None
                for stats in all_stats:
                    if model_name in stats['correct_questions']:
                        model_stats = stats
                        break
                
                if model_stats is None:
                    row.extend(["N/A", 0, 0])
                    continue
                
                if category == 'Overall':
                    total = model_stats['grand_total']
                    correct = model_stats['total_correct'][model_name]
                elif category == 'Other':
                    total = model_stats['grand_total'] - sum(model_stats['total_questions'].get(cat, 0) for cat in model_stats['categories'])
                    correct = model_stats['total_correct'][model_name] - sum(model_stats['correct_questions'][model_name].get(cat, 0) for cat in model_stats['categories'])
                else:
                    total = model_stats['total_questions'].get(category, 0)
                    correct = model_stats['correct_questions'][model_name].get(category, 0)
                
                accuracy = f"{correct/total:.2%}" if total > 0 else "N/A"
                row.extend([accuracy, correct, total])
            
            writer.writerow(row)
    
    print(f"已保存{len(all_model_names)}个模型的准确率统计到: {csv_path}")

def print_accuracy_stats(stats, model_name=None, output_file=None):
    """在终端打印准确率统计，并可输出到文件"""
    output_lines = []
    
    def add_line(line=""):
        nonlocal output_lines
        output_lines.append(line)
        print(line)
    
    categories = stats['categories']
    total_questions = stats['total_questions']
    correct_questions = stats['correct_questions']
    total_correct = stats['total_correct']
    grand_total = stats['grand_total']
    
    # 如果指定了模型名，只打印该模型的统计
    if model_name:
        model_stats = {
            model_name: correct_questions[model_name]
        }
    else:
        model_stats = correct_questions
    
    for name in model_stats:
        add_line(f"\n模型: {name}")
        add_line("-"*40)
        
        # 按类别输出准确率
        for category in categories:
            cat_total = total_questions.get(category, 0)
            cat_correct = model_stats[name].get(category, 0)
            accuracy = f"{cat_correct/cat_total:.2%}" if cat_total > 0 else "N/A"
            add_line(f"{category}: {accuracy} ({cat_correct}/{cat_total})")
        
        # 输出Other类别
        other_total = grand_total - sum(total_questions.get(cat, 0) for cat in categories)
        other_correct = total_correct[name] - sum(model_stats[name].get(cat, 0) for cat in categories)
        other_accuracy = f"{other_correct/other_total:.2%}" if other_total > 0 else "N/A"
        add_line(f"Other: {other_accuracy} ({other_correct}/{other_total})")
        
        # 输出总体准确率
        overall_accuracy = f"{total_correct[name]/grand_total:.2%}"
        add_line(f"\n总体准确率: {overall_accuracy} ({total_correct[name]}/{grand_total})")
    
    # 输出所有模型的汇总统计
    if not model_name and len(model_stats) > 1:
        add_line("\n" + "="*80)
        add_line("所有模型总体准确率比较")
        add_line("="*80)
        
        for name in model_stats:
            overall_accuracy = f"{total_correct[name]/grand_total:.2%}"
            add_line(f"{name}: {overall_accuracy}")
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"已将准确率统计保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='选择题/开放题评测工具')
    parser.add_argument('--input', required=True, nargs='+', help='输入JSON文件路径，可以指定多个')
    parser.add_argument('--output', nargs='+', help='输出JSON文件路径（可选，可以指定多个，与输入文件一一对应）')
    parser.add_argument('--use-gpt', action='store_true', 
                      help='使用GPT进行评测（默认使用正则表达式匹配）')
    parser.add_argument('--max-distance', type=int, default=50,
                      help='正则匹配时答案距离结尾的最大允许字符数（默认50）')
    parser.add_argument('--accuracy-csv', help='保存所有模型准确率统计的CSV文件路径（可选）')
    parser.add_argument('--open-questions', help='开放题JSONL文件路径（可选）')
    args = parser.parse_args()

    # 验证输入输出文件数量是否匹配
    if args.output and len(args.input) != len(args.output):
        raise ValueError("输入文件和输出文件数量必须一致")
    
    # 判断是否为开放题
    is_open_question = 'open' in ' '.join(args.input).lower() or args.open_questions is not None
    
    # 加载开放题ID列表（如果提供）
    open_question_ids = None
    if args.open_questions:
        open_question_ids = load_open_question_ids(args.open_questions)
    
    # 存储所有模型的统计结果
    all_stats = []
    
    # 处理每个输入文件
    for i, input_file in enumerate(args.input):
        print(f"\n处理文件 {i+1}/{len(args.input)}: {input_file}")
        print(f"处理 {'开放题' if is_open_question else '选择题'} 文件")
        if is_open_question and open_question_ids:
            print(f"从{args.open_questions}加载了{len(open_question_ids)}个开放题ID")
        
        # 加载数据
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"已加载{len(data)}个题目")
        except Exception as e:
            print(f"加载文件失败: {e}")
            continue
        
        # 执行评测并统计准确率
        stats = update_answer_correctness(
            data,
            is_open_question=is_open_question,
            use_gpt=args.use_gpt,
            max_distance=args.max_distance,
            open_question_ids=open_question_ids
        )
        
        # 保存结果
        if args.output:
            output_file = args.output[i]
            output_data = stats.get('evaluated_items', data)  # 使用已评估的题目列表或原始数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"已保存{len(output_data)}条评测结果到{output_file}")
        
        # 确定准确率统计的输出文件
        accuracy_txt_file = None
        if args.accuracy_csv:
            base, _ = os.path.splitext(args.accuracy_csv)
            accuracy_txt_file = f"{base}.txt"
        
        # 打印当前文件的准确率统计
        print_accuracy_stats(stats, output_file=accuracy_txt_file)
        
        # 保存统计结果
        all_stats.append(stats)
    
    # 保存所有模型的汇总统计到CSV
    if args.accuracy_csv and all_stats:
        save_accuracy_to_csv(all_stats, args.accuracy_csv)

if __name__ == "__main__":
    main()
