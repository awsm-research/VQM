import pandas as pd
from tokenizers import Tokenizer
from transformers import RobertaTokenizer

df1 = pd.read_csv("../data/cve_fixes/cve_fixes_train.csv")
df2 = pd.read_csv("../data/cve_fixes/cve_fixes_val.csv")
df3 = pd.read_csv("../data/cve_fixes/cve_fixes_test.csv")
df = pd.concat((df1, df2, df3))

#tokenizer = Tokenizer.from_file('./wordlevel_tokenizer/wordlevel.json')    
#tokenizer = RobertaTokenizer.from_pretrained("../SeqTrans/bpe_tokenizer")
tokenizer = RobertaTokenizer.from_pretrained("./bpe_tokenizer")
#tokenizer = RobertaTokenizer.from_pretrained("../SequenceR/tokenizer")

target = df["target"].tolist()
c = 0
for t in target:
    _ids = tokenizer.encode(t)
    if 3 in _ids:
        c += 1
print(c)
