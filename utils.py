import copy
import re


def is_null(o):
    return True if o is None else False


def deepcopy(o):
    return copy.deepcopy(o)


'''
before: abc
after: ABC
'''


def upper(o):
    return o.upper()


'''
before: 'FSM-GFU-Window2012R2-WIN2012R2-172-30-56-123'
after: '03142832567DF2GI2MN2OR2SUW3'
'''


def string_compression(s):
    s = upper(s)
    s = re.sub(r'[^\w]', '', s)
    s = sorted(s)

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
