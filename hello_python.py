#!/usr/bin/env python3
"""A minimal Hello, World! program to confirm Python execution works."""

def main():
    # Python uses indentation (not braces) for code blocks
    name = "World"
    print(f"Hello, {name}!")
    print("Python execution works correctly.")

    # Demonstrate dynamic typing: the same variable can hold different types
    x = 42          # x is an integer
    print(f"x = {x}, type: {type(x).__name__}")
    x = "changed"   # x is now a string
    print(f"x = {x}, type: {type(x).__name__}")
    x = [1, 2, 3]   # x is now a list
    print(f"x = {x}, type: {type(x).__name__}")

if __name__ == "__main__":
    main()
