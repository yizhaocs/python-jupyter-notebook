# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def solve(s):
   import re
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s = "FSM-GFU-Window2012R2-WIN2012R2-172-30-56-123"
    print(solve(s))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
