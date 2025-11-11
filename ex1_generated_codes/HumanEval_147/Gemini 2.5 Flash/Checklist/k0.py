def get_max_triples(n):
    # Create the array a based on the given formula
    a = [i * i - i + 1 for i in range(1, n + 1)]
    
    # Initialize counters for each remainder when divided by 3
    count_remainder_0 = 0
    count_remainder_1 = 0
    count_remainder_2 = 0
    
    # Count occurrences of each remainder
    for num in a:
        if num % 3 == 0:
            count_remainder_0 += 1
        elif num % 3 == 1:
            count_remainder_1 += 1
        else:
            count_remainder_2 += 1
    
    # Calculate the number of valid triples
    # (a[i] + a[j] + a[k]) % 3 = 0
    # Possible combinations are:
    # 0 + 0 + 0, 1 + 1 + 1, 2 + 2 + 2,
    # 0 + 1 + 2
    count_triples = (
        count_remainder_0 * (count_remainder_0 - 1) * (count_remainder_0 - 2) // 6 +
        count_remainder_1 * (count_remainder_1 - 1) * (count_remainder_1 - 2) // 6 +
        count_remainder_2 * (count_remainder_2 - 1) * (count_remainder_2 - 2) // 6 +
        count_remainder_0 * count_remainder_1 * count_remainder_2
    )
    
    return count_triples