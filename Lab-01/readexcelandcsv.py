import pandas as pd

# Read the CSV file from your computer
winedatac = pd.read_csv(r"C:\Users\nickk\Downloads\ml\Lab-01\wine.csv")

# Print the entire dataset
print(winedatac)

# Display the first 5 rows of the dataset
print(winedatac.head())

# Display the number of rows and columns
print("Shape\n", winedatac.shape)

# Display the column names
print("Columns\n", winedatac.columns)

# Display the data type of each column
print("Data Types\n", winedatac.dtypes)

# Display the number of dimensions (DataFrame = 2D)
print("Dimensions\n", winedatac.ndim)

# Display the total number of elements (rows × columns)
print("Size\n", winedatac.size)

# Read the Excel file
# Make sure the file exists at this location
winedatae = pd.read_excel(r"C:\Users\nickk\Downloads\ml\Lab-01\wine.xls")

print("\n")

# Print the Excel dataset
print(winedatae)