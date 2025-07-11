import re
import os
import random


def compute_score(solution_str, ground_truth, method='strict'):
    """The scoring function for critique verification.

    Args:
        solution_str: the solution text containing critique conclusion
        ground_truth: the expected conclusion (e.g., "right" or "wrong")
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        
    Returns:
        dict: {"score": float, "correctness": bool}
        - score: combined score for both conclusion correctness and format correctness
        - correctness: whether the conclusion is correct (True/False)
    """
    # 提取结论模式
    conclusion_text, conclusion_is_correct, has_conclusion_format = extract_conclusion_pattern(solution_str)
    
    # 判断结论是否正确（与ground_truth比较）
    # ground_truth应该是"right"或"wrong"
    expected_correct = ground_truth.lower() in ['right', 'correct']
    conclusion_correct = conclusion_is_correct == expected_correct

    reward_function_type = str(os.environ.get('REWORD_FUNCTION_TYPE', "mix"))
    format_penalty_value = float(os.environ.get('FORMAT_PENALTY_VALUE', "-1"))

    print(f"Reward function type: {reward_function_type}")
    print(f"Format penalty value: {format_penalty_value}")
    # 计算分数
    if reward_function_type == 'mix':
        if conclusion_correct:
            score = 1.0
        else:
            score = 0.0
    elif reward_function_type == 'independent':
        if conclusion_correct and has_conclusion_format:
            score = 1.0
        elif conclusion_correct and not has_conclusion_format:
            score = 0.5
        elif not conclusion_correct and has_conclusion_format:
            score = -0.5
        else:
            score = format_penalty_value
    else:
        raise ValueError(f"Invalid reward function type: {reward_function_type}")
            
    if random.random() < 0.05:
        # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth Conclusion]\n{ground_truth}")
        print(f"\n[Conclusion Pattern]\n{conclusion_text}")
        print(f"\n[Has Conclusion Format]\n{has_conclusion_format}")
        print(f"\n[Extracted Conclusion Is Correct]\n{conclusion_is_correct}")
        print(f"\n[Expected Conclusion Is Correct]\n{expected_correct}")
        print(f"\n[Conclusion Correct]\n{conclusion_correct}")
        print(f"\n[Combined Score]\n{score}")
    
    return {"score": score, "correctness": conclusion_correct}


def extract_conclusion_pattern(text):
    """
    提取文本中的结论模式，如 "\\n\\nConclusion: right [END]\\n\\n"

    Args:
        text: 输入文本

    Returns:
        tuple: (conclusion_text, is_correct, has_conclusion_format)
        - conclusion_text: 提取的结论文本
        - is_correct: 根据结论判断是否正确 (True/False)
        - has_conclusion_format: 是否包含结论格式 (True/False)
    """
    # 匹配结论模式，支持大小写不敏感
    pattern = r'\nConclusion:\s*(right|wrong|correct|incorrect)\s*\[END\]\n'
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        conclusion_word = match.group(1).lower()
        # 判断是否正确：right 或 correct 表示正确
        is_correct = conclusion_word in ['right', 'correct']
        return match.group(0), is_correct, True
    else:
        return None, False, False


if __name__ == "__main__":
    # 测试新的结论提取逻辑
    test_cases = [
        {
            "name": "Correct conclusion with right",
            "text": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: right [END]\n\n",
            "expected_correct": True,
            "expected_has_format": True
        },
        {
            "name": "Correct conclusion with correct",
            "text": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: correct [END]\n\n",
            "expected_correct": True,
            "expected_has_format": True
        },
        {
            "name": "Incorrect conclusion with wrong",
            "text": "The student's solution has errors.\n\nConclusion: wrong [END]\n\n",
            "expected_correct": False,
            "expected_has_format": True
        },
        {
            "name": "Incorrect conclusion with incorrect",
            "text": "The student's solution has errors.\n\nConclusion: incorrect [END]\n\n",
            "expected_correct": False,
            "expected_has_format": True
        },
        {
            "name": "Case insensitive test",
            "text": "Some text.\n\nConclusion: RIGHT [END]\n\n",
            "expected_correct": True,
            "expected_has_format": True
        },
        {
            "name": "No conclusion format",
            "text": "Just some regular text without conclusion format.",
            "expected_correct": False,
            "expected_has_format": False
        }
    ]
    
    print("Testing extract_conclusion_pattern function:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {repr(test_case['text'])}")
        
        conclusion_text, is_correct, has_format = extract_conclusion_pattern(test_case['text'])
        
        print(f"Extracted conclusion: {repr(conclusion_text)}")
        print(f"Is correct: {is_correct}")
        print(f"Has format: {has_format}")
        print(f"Expected correct: {test_case['expected_correct']}")
        print(f"Expected has format: {test_case['expected_has_format']}")
        
        if is_correct == test_case['expected_correct'] and has_format == test_case['expected_has_format']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
    
    print("\n" + "=" * 50)
    print("Testing compute_score function:")
    
    # 测试compute_score函数 - 新的逻辑
    test_cases_score = [
        {
            "name": "Correct conclusion with right ground truth",
            "solution": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: right [END]\n\n",
            "ground_truth": "right",
            "expected_correctness": True,
            "expected_score": 1.0
        },
        {
            "name": "Correct conclusion with wrong ground truth",
            "solution": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: right [END]\n\n",
            "ground_truth": "wrong",
            "expected_correctness": False,
            "expected_score": 0.0
        },
        {
            "name": "Wrong conclusion with wrong ground truth",
            "solution": "The student's solution has errors.\n\nConclusion: wrong [END]\n\n",
            "ground_truth": "wrong",
            "expected_correctness": True,
            "expected_score": 1.0
        },
        {
            "name": "No conclusion format with right ground truth",
            "solution": "Just some regular text without conclusion format.",
            "ground_truth": "right",
            "expected_correctness": False,
            "expected_score": 0.0
        }
    ]
    
    for i, test_case in enumerate(test_cases_score, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Solution: {repr(test_case['solution'])}")
        print(f"Ground truth: {test_case['ground_truth']}")
        
        result = compute_score(test_case['solution'], test_case['ground_truth'])
        
        print(f"Result: {result}")
        print(f"Expected correctness: {test_case['expected_correctness']}")
        print(f"Expected score: {test_case['expected_score']}")
        
        if (result['correctness'] == test_case['expected_correctness'] and 
            abs(result['score'] - test_case['expected_score']) < 0.001):
            print("✅ PASS")
        else:
            print("❌ FAIL")