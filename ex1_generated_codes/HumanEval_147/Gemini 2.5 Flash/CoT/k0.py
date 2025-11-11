def get_max_triples(n):
    # Step 1: Generate the array a
    a = [i * i - i + 1 for i in range(1, n + 1)]
    
    # Step 2: Count occurrences of each residue modulo 3
    count_0 = sum(1 for x in a if x % 3 == 0)
    count_1 = sum(1 for x in a if x % 3 == 1)
    count_2 = sum(1 for x in a if x % 3 == 2)
    
    # Step 3: Calculate the number of valid triples
    # Case 1: All three numbers have residue 0
    case_1 = count_0 * (count_0 - 1) * (count_0 - 2) // 6
    
    # Case 2: All three numbers have residue 1
    case_2 = count_1 * (count_1 - 1) * (count_1 - 2) // 6
    
    # Case 3: All three numbers have residue 2
    case_3 = count_2 * (count_2 - 1) * (count_2 - 2) // 6
    
    # Case 4: One number has residue 0, one has residue 1, and one has residue 2
    case_4 = count_0 * count_1 * count_2
    
    # Total valid triples
    total_triples = case_1 + case_2 + case_3 + case_4
    
    return total_triples

# Example usage:
print(get_max_triples(5))  # Output: 1