import pandas as pd

# df = pd.read_csv("Cases-All 2018-2021 Cases.csv")


# df = df[['Index', 'Opinion year', 'Alleged error', 'Allegation graf', 'Court Holding', 'Court holding graf']]
# print(df.head)
#
# df['Alleged error'] = df['Alleged error'].str.split(',')
#
#
# df_exp = df.explode('Alleged error')
# print(df_exp.head)
# df_exp.to_csv("all-expanded-aoe-2018-2021.csv", index=False)
#
# index_counts = df_exp.index.value_counts()
# multiple_indices = index_counts[index_counts > 1].index
# df_filtered = df_exp[df_exp.index.isin(multiple_indices)]
#
# print(f"\nIndices that appear multiple times: {multiple_indices.tolist()}")
# print("Filtered DataFrame (only multiple entries):")
# print(df_filtered.head)
# df_filtered.to_csv("expanded-aoe-2018-2021.csv", index=False)

df_multiples = pd.read_excel("expanded-aoe-2018-2021_LES.xlsx")
replacements = {
    'â€™': "'",  # apostrophe
    'â€œ': '"',  # left double quote
    'â€': '"',   # right double quote
    'â€"': '—',  # em dash
    'â€"': '–',  # en dash
    'Â': '',     # non-breaking space artifacts
}

# Apply replacements to all text columns
for col in df_multiples.select_dtypes(include=['object']).columns:
    df_multiples[col] = df_multiples[col].astype(str)
    for old, new in replacements.items():
        df_multiples[col] = df_multiples[col].str.replace(old, new, regex=False)

df_all = pd.read_csv("all-expanded-aoe-2018-2021.csv")
df_filtered = pd.read_csv("expanded-aoe-2018-2021.csv")

df_singles = pd.concat([df_all, df_filtered]).drop_duplicates(keep=False)

print(df_singles.head)
print(df_multiples.head)
df = pd.concat([df_singles, df_multiples], axis=0)

df.sort_values('Index')

df.to_csv("labeled-aoe-2018-2021.csv", index=False)
