# Task 1 Report
## `flatten_list`
The function `flatten_list` takes a nested list and returns a flat list.

A simple implementation of the function is to use **recursion** to traverse all elements in the nested list and append them to a flat list. The implementation is as follows:
```python
def flatten_list(nested_list: list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list
```

Another interesting approach is to use **string methods** to convert the list to a string, remove the brackets, and split the string to get the flat list. The implementation is as follows:
```python
def flatten_list(nested_list: list):
    nested_str = str(nested_list)
    flattened_str = nested_str.replace('[', '').replace(']', '')
    flat_list = [int(x) for x in flattened_str.split(',') if x.strip()]
    return flat_list
```

The proformance of these two implementations on data of different scales is shown in the following table:

| Methods | $10^3$ | $10^4$ | $10^5$ | $10^6$ | $10^7$ |
|---------|--------|--------|--------|--------|--------|
| Recursion | 0.0000s | 0.0014s | 0.01951s | 0.2227s | 2.2643s |
| String Methods | 0.0000s | 0.0040s | 0.0290s | 0.3867s | 3.9976s |


## `char_count`

The function `char_count` takes a string and returns a dictionary with the count of each character in the string.

The implementation of `char_count()` is quite simple; the **built-in function** `str.count()` in Python allows for easy retrieval of the result. The implementation is as follows:
```python
def char_count(string: str):
    return {char: s.count(char) for char in set(s)}
```

Of course, we can also use a more **traditional approach** to count the characters in the string with a complexity of $O(N)$. The implementation is as follows:
```python
def char_count(string: str):
    char_dict = {}
    for char in string:
        if char in char_dict:
            char_dict[char] += 1
        else:
            char_dict[char] = 1
    return char_dict
```

The performance of these two implementations on data of different scales is shown in the following table:

| Methods | $10^3$ | $10^4$ | $10^5$ | $10^6$ | $10^7$ |
|---------|--------|--------|--------|--------|--------|
|built-in| 0.0000s| 0.0000s| 0.0025s| 0.0210s| 0.2049s|
|traditional| 0.0000s| 0.0009s| 0.061s| 0.0594s| 0.6135s|

