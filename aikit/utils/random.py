import random as r
import string as s


def number(a=10, b=100): return r.randint(a, b)


choice = r.choice


def string(length=5, chars=None):
    if not chars:
        chars = s.ascii_letters + s.digits + s.punctuation

    return ''.join(choice(chars) for _ in range(length))


def strings(count=number(2, 10)):
    return [string() for _ in range(count)]


def shape():
    return tuple([number(2, 5) for _ in range(number(2, 4))])
