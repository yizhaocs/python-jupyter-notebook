import random
import string
from collections import defaultdict


class StringShortenTool:
    codeDB, urlDB = defaultdict(), defaultdict()
    chars = string.ascii_letters + string.digits

    def getCode(self) -> str:
        code = ''.join(random.choice(self.chars) for i in range(6))
        return code

    def encode(self, longUrl: str) -> str:
        if longUrl in self.urlDB: return self.urlDB[longUrl]
        code = self.getCode()
        while code in self.codeDB: code = self.getCode()
        self.codeDB[code] = longUrl
        self.urlDB[longUrl] = code
        return code

    def decode(self, shortUrl: str) -> str:
        return self.codeDB[shortUrl]


if __name__ == '__main__':
    my_code = StringShortenTool()
    encode = my_code.encode(longUrl='a-d-c-d-172.3.4.1')
    print(f"encode:{encode}")
    decode = my_code.decode(shortUrl=encode)
    print(f"decode:{decode}")
