def below_threshold(l: list, t: int) -> bool:
    return all(x < t for x in l)

# Test cases
assert below_threshold([1, 2, 4, 10], 100) == True
assert below_threshold([1, 20, 4, 10], 5) == False