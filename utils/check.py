from sympy import symbols, simplify, Rational
import re

def parse_latex_vector(latex_str):
    # 去除左括号和右括号以及反斜杠
    clean_str = latex_str.replace("\\left(", "").replace("\\right)", "")
    
    # 替换LaTeX的\frac{}{}为可以计算的形式
    frac_pattern = r'\\frac\{(\d+)\}\{(\d+)\}'
    clean_str = re.sub(frac_pattern, r'\1/\2', clean_str)
    
    # 分割字符串得到各个元素
    elements = clean_str.split(',')
    
    # 解析每个元素
    parsed_elements = []
    for el in elements:
        el = el.strip()  # 移除可能的额外空格
        try:
            # 检查是否为整数或小数
            num = float(el)
        except ValueError:
            # 如果不是，则尝试作为Rational对象解析
            num = Rational(el).evalf()
        parsed_elements.append(num)
    
    return tuple(parsed_elements)

def strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    string = fix_a_slash_b(string)
    string = string.replace('x \\in', '').strip()  # noqa: W605

    # a_b == a, a_{b} == a_b for bit conversion
    if string.find('_') >= 0:
        p = string.split('_')
        p[1] = p[1].replace('{', '').replace('}', '')
        string = '_'.join(p)

    # 10800 == 10,800; we only deal with single number
    if string.strip().find(' ') == -1 and string.find('(') == -1:
        string = string.replace(',', '')

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # print("WARNING: Both None")
        return False
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def normalize_number(text):
        if not text:
            return None
        # first remove all non-digit characters except decimal point and minus sign
        text = re.sub(r'[^\d.-]', '', text.strip())
        # then match the cleaned number
        match = re.match(r'^-?\d*\.?\d+$', text)
        if not match:
            return None
        try:
            # convert to float and then back to string to normalize the format
            return str(float(match.group()))
        except ValueError:
            return None

def extract_answer(text):
    """
    extract the answer after ####, only keep numbers
    """
    if "####" in text:
        # use split to get the part after ####
        answer = text.split("####")[1].strip()
        return answer
    else:
        return None


def check_answer_gsm8k(prediction: str, label: str) -> bool:
    """
    Evaluate the accuracy of GSM8K answer
    Args:
        prediction: The predicted answer with reasoning
        label: The correct answer
    Returns:
        bool: Whether the prediction matches the label
    """
    cot_answer = extract_answer(prediction)
    label_answer = extract_answer(label)
    # extract the number from the answer
    cot_answer = normalize_number(cot_answer)
    label_answer = normalize_number(label_answer)
    return cot_answer == label_answer

def check_answer_math(prediction: str, label: str) -> bool:
    """
    Evaluate the accuracy of Math answer
    """ 
    cot_answer = extract_answer(prediction)
    
    # 如果无法提取答案，直接返回False
    if not cot_answer:
        return False
    
    # 直接字符串比较
    if str(cot_answer) == str(label):
        return True
    
    # 标准化数字比较
    if str(normalize_number(cot_answer)) == str(normalize_number(label)):
        return True
    
    # 向量比较
    try:
        import ast
        label_tuple = parse_latex_vector(label)
        pred_tuple = ast.literal_eval(cot_answer)  # 使用更安全的ast.literal_eval
        # 如果label_tuple是单元素元组，提取第一个元素进行比较
        if len(label_tuple) == 1:
            if label_tuple[0] == pred_tuple:
                return True
        elif label_tuple == pred_tuple:
            return True
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # 符号表达式比较
    try:
        if 'x' in label and 'x' in cot_answer:
            # 处理包含变量的表达式
            label_processed = label.replace('x', '*x').replace(' ', '')
            cot_answer_processed = cot_answer.replace('x', '*x').replace(' ', '')
            if simplify(label_processed) == simplify(cot_answer_processed):
                return True
        elif simplify(label) == simplify(cot_answer):
            return True
    except (ValueError, TypeError, AttributeError):
        pass
    
    # 数值比较
    try:
        if float(label) == float(cot_answer):
            return True
    except (ValueError, TypeError):
        pass

    # 确定答案类型
    if label.isdigit():
        answer_type = 'digit'
    elif label.isupper() and label.isalpha():
        answer_type = 'option'
    elif label.lower() in ['yes','no']:
        answer_type = 'yesorno'
        label_normalized = label.lower()  # 不修改原始label
    else:
        answer_type = 'formula'

    # 根据类型处理预测答案
    if answer_type == 'option':
        cot_answer = cot_answer.strip()[0]
    elif answer_type == 'yesorno':
        cot_answer = cot_answer.lower()
    elif answer_type == 'formula':
        cot_answer = cot_answer.replace('$','')
    
    # 使用is_equiv进行最终比较
    if answer_type == 'yesorno':
        return is_equiv(label_normalized, cot_answer)
    else:
        return is_equiv(label, cot_answer)

def check_answer_gsm_hard(prediction: str, label: str) -> bool:
    """
    Evaluate the accuracy of GSM-Hard answer
    """
    cot_answer = extract_answer(prediction)
    cot_answer = normalize_number(cot_answer)
    if cot_answer is None:
        return False
    else:
        cot_answer = float(cot_answer)
    return cot_answer == label

def check_answer_strategyqa(prediction: str, label: bool) -> bool:
    """
    Evaluate the accuracy of StrategyQA answer
    Args:
        prediction: The predicted answer with reasoning
        label: The correct answer (boolean)
    Returns:
        bool: Whether the prediction matches the label
    """
    cot_answer = extract_answer(prediction)
    if not cot_answer:
        return False
        
    cot_answer_normalized = cot_answer.strip().lower()
    if cot_answer_normalized in ['true', 'yes', '1']:
        cot_answer_bool = True
    elif cot_answer_normalized in ['false', 'no', '0']:
        cot_answer_bool = False
    else:
        cot_answer_bool = None
    
    return cot_answer_bool == label

def check_answer_bamboogle(prediction: str, label: bool) -> bool:
    """
    Evaluate the accuracy of Bamboogle answer
    Args:
        prediction: The predicted answer with reasoning
        label: The correct answer (string)
    Returns:
        bool: Whether the prediction matches the label
    """
    cot_answer = extract_answer(prediction)
    if not cot_answer:
        return False
        
    cot_answer_normalized = cot_answer.strip().lower()
    label_normalized = label.strip().lower()

    if label_normalized in cot_answer_normalized:
        return True
    else:
        return False

def check_answer_coin(prediction: str, label: str) -> bool:
    """
    Evaluate the accuracy of Coin answer
    """
    cot_answer = extract_answer(prediction)
    if not cot_answer:
        return False
        
    cot_answer_normalized = cot_answer.strip().lower()
    
    return cot_answer_normalized == label

def check_answer_letter(prediction: str, label: str) -> bool:
    """
    Evaluate the accuracy of Letter answer
    """
    cot_answer = extract_answer(prediction)
    if not cot_answer:
        return False
    normalized_answer = normalize_text(label, stem=False)
    normalized_gen = normalize_text(cot_answer, stem=False)
    return normalized_answer == normalized_gen

def check_answer_legal(prediction: str, label: str) -> bool:
    """
    Evaluates exact match using balanced_accuracy.
    """
    cot_answer = extract_answer(prediction)
    if not cot_answer:
        return False
    normalized_answer = normalize_text(label, stem=False)
    normalized_gen = normalize_text(cot_answer, stem=False)
    return normalized_answer == normalized_gen

def check_answer_Headline(prediction: str, label: str) -> bool:
    """
    Evaluate the accuracy of Headline answer
    """
    cot_answer = extract_answer(prediction)
    if not cot_answer:
        return False
    label_normalized = label.strip().lower()
    cot_answer_normalized = cot_answer.strip().lower()
    return cot_answer_normalized == label_normalized

def normalize_text(text: str, stem: bool) -> str:
    """
    Normalizes strings.

    Args:
        - text: text to normalize
        - stem: whether or not to apply a stemmer
    
    Returns: normalized text
    """
    import string
    # Remove punctuation
    text = str(text).translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = text.strip()

    # Make lower case
    text = text.lower()

    # Stem
    if stem:
        from nltk.stem.porter import PorterStemmer
        text = PorterStemmer().stem(text)
    return text