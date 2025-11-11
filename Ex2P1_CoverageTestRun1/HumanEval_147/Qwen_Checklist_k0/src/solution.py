def get_max_triples(n):
    # Initialize the array a with values based on the given formula
    a = [i * i - i + 1 for i in range(1, n + 1)]
    
    # Count the number of valid triples (a[i], a[j], a[k]) where i < j < k and their sum is a multiple of 3
    count = 0
    
    # Iterate over all possible triplets (i, j, k) with i < j < k
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    count += 1
    
    return count