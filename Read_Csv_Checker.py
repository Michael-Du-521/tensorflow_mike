import pandas as pd

# 定义训练必要的参数
folder = "D:/AI_Sample_06152023/Adjoin"
fileName = '/Result.txt'

print(folder+fileName)
result = pd.read_csv(folder + fileName, sep='\t', header=0)