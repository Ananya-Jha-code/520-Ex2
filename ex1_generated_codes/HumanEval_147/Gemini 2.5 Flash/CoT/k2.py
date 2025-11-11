def get_max_triples(n):
    # Step 1: Generate the array a
    a = [i * i - i + 1 for i in range(1, n + 1)]
    
    # Step 2: Count valid triples (a[i], a[j], a[k]) where i < j < k and a[i] + a[j] + a[k] is a multiple of 3
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    count += 1
    
    # Step 3: Return the count
    return count

# Example usage:
print(get_max_triples(5))  # Output: 1