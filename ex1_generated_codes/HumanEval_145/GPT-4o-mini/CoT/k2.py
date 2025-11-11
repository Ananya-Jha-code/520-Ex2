def order_by_points(nums):  
    return [item for pair in sorted((value, i) for (i, value) in enumerate(nums))]
    
print("Testing with the given example...")
assert order_by_points([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11], "Failed!"   #[-1,-11,1,-12,11] for sorted by sum of digits and index based on original list
print("All tests passed!")