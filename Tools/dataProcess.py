# import pandas as pd

# # Upload the CSV file
# df = pd.read_csv(r'lang\ben.txt', sep='\t')



# # Remove the last column
# df = df.iloc[:, :-1]

# # Save the cleaned CSV as "cleaned_csv"
# df.to_csv('lang\cleaned_ben.tsv', sep='\t', index=False)


# if __name__ == "__main__":
#     print(df.head())


import pandas as pd

# Upload the CSV file
df = pd.read_csv(r'lang\cleaned_ben.tsv', sep='\t')

# Swap the values in the first two columns
df.iloc[:, [0, 1]] = df.iloc[:, [1, 0]].values

# Save the modified CSV as "cleaned_ben.tsv"
df.to_csv(r'lang\cleaned_ben.tsv', sep='\t', index=False)

if __name__ == "__main__":
    print(df.head())


