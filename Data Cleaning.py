import pandas as pd

df = pd.read_excel(r'/Users/Suli/Documents/Python Practice/Datasets/Customer Call List.xlsx')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Initial clean up
df.drop_duplicates(inplace=True)
df.drop(columns='Not_Useful_Column', inplace=True)

# Last_Name column clean up.
df['Last_Name'] = df['Last_Name'].str.strip('./_')

# Phone number clean up
df['Phone_Number'] = df['Phone_Number'].str.replace('[^a-zA-Z0-9]', '', regex=True)
df['Phone_Number'] = df['Phone_Number'].apply(lambda x: str(x))
df['Phone_Number'] = df['Phone_Number'].apply(lambda x: x[0:3] + '-' + x[3:6] + '-' + x[6:10])
df['Phone_Number'] = df['Phone_Number'].str.replace('nan--', '').str.replace('Na--', '')

# Clean up address, split it into different columns
df[['Street_Address', 'State', 'Zip_Code']] = df['Address'].str.split(',', n=2, expand=True)
df = df.drop(columns='Address')

# Clean up Paying Customer column
df['Paying Customer'] = df['Paying Customer'].str.replace('Yes', 'Y')
df['Paying Customer'] = df['Paying Customer'].str.replace('No', 'N')

df['Do_Not_Contact'] = df['Do_Not_Contact'].str.replace('Yes', 'Y')
df['Do_Not_Contact'] = df['Do_Not_Contact'].str.replace('No', 'N')

df = df.replace('N/a', '').fillna('').replace('Y', 'Yes').replace('N', 'No')
print(df)

