def recursive(n):
    if n == 3:
        return
    print(f"n before : {n}")
    recursive(n-1)
    print(f"n after : {n}")

recursive(6)