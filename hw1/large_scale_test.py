import time
import random
from submission import *

# Generate a random nested list with a given depth and length
def generate_random_nested_list(depth, length):
    if depth == 1:
        return [random.randint(0, 100) for _ in range(length)]
    return [generate_random_nested_list(depth - 1, length) for _ in range(length)]

# Gradually increase the input scale and see the time change
for input_scale in [1e2]:
    begin = time.time()
    # Generate the random input data which will be nested in random depth
    nested_list = generate_random_nested_list(3, 10)
    print(nested_list)