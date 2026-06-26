import pandas as pd

# Read CSV file
winedatac = pd.read_csv(r"C:\Users\nickk\Downloads\ml\Lab-03\wine.csv")

print(winedatac)
print(winedatac.head())
print("shape\n", winedatac.shape)
print("columns\n", winedatac.columns)
print("dtypes\n", winedatac.dtypes)
print("ndim\n", winedatac.ndim)
print("size\n", winedatac.size)

# Read Excel file
winedatae = pd.read_excel(r"C:\Users\nickk\Downloads\ml\Lab-03\wine.xls")  # Change to wine.xls if needed

print("\n")

print(winedatae)
print(winedatae.head())
print("shape\n", winedatae.shape)
print("columns\n", winedatae.columns)
print("dtypes\n", winedatae.dtypes)
print("ndim\n", winedatae.ndim)
print("size\n", winedatae.size)