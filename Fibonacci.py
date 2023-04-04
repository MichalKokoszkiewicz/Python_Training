def gen_fib():
    yield 0
    yield 1
    v1, v2 = 0, 1
    while True:
        yield v1 + v2
        v1, v2 = v2, v1 + v2

fib = gen_fib()
print(next(fib))
print(next(fib))
print(next(fib))

for i in range(10):
    print(next(fib))

print(next(fib))
print(next(fib))
print(next(fib))
