import copy
import re
import time


def thread_sleep(second):
    time.sleep(second)

def object_is_null(o):
    return True if o is None else False

def object_is_not_null(o):
    return False if o is None else True

def object_deepcopy(o):
    return copy.deepcopy(o)

'''
    before: dict = {"x":"123", "y":"", "z": "456"}
    after: {'x': '123', 'z': '456'}
'''
def dict_remove_keys_with_empty_strings(dict):
    return {k: v for k, v in dict.items() if v}

'''
before: abc
after: ABC
'''


def string_upper(o):
    return o.upper()

'''
before: ABC
after: abc
'''


def string_lower(o):
    return o.lower()

'''
before: a b c
after: abc
'''
def string_remove_space(s):
    return s.replace(" ", "")

'''
before: '123-daxac'
after: ['-', '1', '2', '3', 'a', 'a', 'c', 'd', 'x']
'''
def string_sorted(s):
    return sorted(s)
'''
before: 'FSM-GFU-Window2012R2-WIN2012R2-172-30-56-123'
after: '03142832567DF2GI2MN2OR2SUW3'
'''


def string_compression(s):
    s = string_upper(s)
    s = re.sub(r'[^\w]', '', s)
    s = string_sorted(s)

    res = ""
    cnt = 1
    for i in range(1, len(s)):
        if s[i - 1] == s[i]:
            cnt += 1
        else:
            res = res + s[i - 1]
            if cnt > 1:
                res += str(cnt)
            cnt = 1
    res = res + s[-1]
    if cnt > 1:
        res += str(cnt)
    return res

def epoch_time():
    import time
    return int(time.time() * 1000)


if __name__ == '__main__':
    s = "123 daxac"
    print(string_remove_space(s))
    dict = {"x":"123", "y":"", "z": "456"}
    print(dict_remove_keys_with_empty_strings(dict))
