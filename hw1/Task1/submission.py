# def flatten_list(nested_list: list):
#     flat_list = []
#     for item in nested_list:
#         if isinstance(item, list):
#             flat_list.extend(flatten_list(item))  # 递归展开
#         else:
#             flat_list.append(item)  # 添加非列表元素
#     return flat_list

def flatten_list(nested_list: list):
    # 将嵌套列表转换为字符串
    nested_str = str(nested_list)
    # 去掉中括号
    flattened_str = nested_str.replace('[', '').replace(']', '')
    # 将字符串转换为整数列表，去掉多余的空格
    flat_list = [int(x) for x in flattened_str.split(',') if x.strip()]
    return flat_list

# def flatten_list(nested_list):
#     flat_list = []

#     def flatten(item):
#         if isinstance(item, list):
#             for sub_item in item:
#                 flatten(sub_item)  
#         else:
#             flat_list.append(item)  

#     flatten(nested_list)  
#     return flat_list

# def char_count(s: str):
#     return {char: s.count(char) for char in set(s)}

def char_count(string: str):
    char_dict = {}
    for char in string:
        if char in char_dict:
            char_dict[char] += 1
        else:
            char_dict[char] = 1
    return char_dict