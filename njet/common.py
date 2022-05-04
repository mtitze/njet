def check_zero(value):
    # check if a value is zero; value may be an iterable
    check = value == 0
    if hasattr(check, '__iter__'):
        return all(check)
    else:
        return check
    
def factorials(n: int):
    k = 1
    facts = [1]
    for j in range(1, n + 1):
        k *= j
        facts.append(k)
    return facts

def n_over_ks(n: int):
    facts = factorials(n)
    return [[facts[j]//(facts[k]*facts[j - k]) for k in range(j + 1)] for j in range(len(facts))]
