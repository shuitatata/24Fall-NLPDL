import random
import time
from submission import *

def generate_nested_list(depth, elements_counts):
    # Recursive boundary condition
    if elements_counts == 1:
        return [random.randint(0, 100)]

    # Random number of groups
    x = random.randint(1, elements_counts)  # At least 1 group, at most elements_counts groups

    # Distribute elements using the divider method
    if x == 1:
        groups_sizes = [elements_counts]  # Only one group, size is elements_counts
    else:
        splits = sorted(random.sample(range(1, elements_counts), x - 1))
        groups_sizes = [splits[0]] + [splits[i] - splits[i - 1] for i in range(1, len(splits))] + [elements_counts - splits[-1]]

    nested_list = []
    for group_size in groups_sizes:
        # Randomly assign depth for the current group
        current_depth = random.randint(0, depth - 1)
        
        # Return random integers at the bottom level (depth is 0)
        if current_depth == 0:
            for _ in range(group_size):
                nested_list.append(random.randint(0, 100))
        else:
            # Recursively generate the nested list for the current group
            nested_list.append(generate_nested_list(current_depth, group_size))

    return nested_list

def judge_flatten_list(flat_list):
    # Check if the flattened list is correct
    for item in flat_list:
        if isinstance(item, list):
            return False
    return True

if __name__ == "__main__":
    # Input scale list
    scale_list = [1e3, 1e4, 1e5, 1e6, 1e7]
    test_id = 1

    if test_id == 0:
        # Test the performance of the flatten_list function
        for input_scale in scale_list:
            # Generate a nested list with the specified scale
            nested_list = generate_nested_list(input_scale//10, int(input_scale))

            # Start timing
            start_time = time.time()
            flatten_list(nested_list)
            end_time = time.time()

            # Judge the correctness of the flatten_list function
            if not judge_flatten_list(flatten_list(nested_list)):
                print("The flatten_list function is incorrect!")
                break

            # Output the time cost
            print(f"Input scale: {input_scale}, Time cost: {end_time - start_time:.8f}s")

    if test_id == 1:
        # Test the performance of the char_count function
        for input_scale in scale_list:
            # Generate a random string with the specified scale
            s = ''.join([chr(random.randint(97, 122)) for _ in range(int(input_scale))])

            # Start timing
            start_time = time.time()
            char_count(s)
            end_time = time.time()

            # Output the time cost
            print(f"Input scale: {input_scale}, Time cost: {end_time - start_time:.8f}s")