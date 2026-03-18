# crafting_interpreters_py

A toy interpreter for a tiny custom language.

## How to start
```
fun fib(n) {
    if (n <= 1) {
        return n;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

print fib(10);
```

Store source code in file (like test.src), and execute following command:
> poetry run run -f test.src

Or use following command and try interpreter in a REPL style:
> poetry run run -p
