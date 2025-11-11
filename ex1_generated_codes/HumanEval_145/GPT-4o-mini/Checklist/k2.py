def order_by_points(nums):  
    return [item for pair in sorted((value, i) for i, value in enumerate(nums))]
    
print("Testing with positive numbers:")
assert order_by_points([1024579836, 1]) == [] # Empty list should return empty array.  
assert order_by_points([-1,-11,-12,-123,-1234])==[-1 , -11, -12, -123, -1234]# Negative numbers sorted by sum of digits and then index in list.
assert order_by_points([567890, 123456,-1-2])==[-(*) , (*)] # Testing with negative integers (negative sign at start).  
print("All tests passed!")