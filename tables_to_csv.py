import pandas as pd

df = pd.read_feather(r'datasets/validate.df')



del df['index']

print(df)

df.to_csv('validate.csv')

bd = pd.read_csv('test.csv', index_col=0)


print(bd)


