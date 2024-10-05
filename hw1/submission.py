def flatten_list(nested_list: list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list
def char_count(s: str):
    return {char: s.count(char) for char in set(s)}
    