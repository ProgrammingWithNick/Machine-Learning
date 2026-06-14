from sklearn.preprocessing import LabelEncoder

# Sample ordinal data
ordinal_data = ['L', 'M', 'M', 'XL', 'XL']

# Define the order of the categories
ordinal_categories = ['XL', 'L', 'M']

# Create a LabelEncoder object with specified order
label_encoder = LabelEncoder()
label_encoder.fit(ordinal_categories)

# Transform the ordinal data using label encoding
encoded_ordinal_data = label_encoder.transform(ordinal_data)

# Display the original and encoded ordinal data
print("Original ordinal data:", ordinal_data)
print("Encoded ordinal data:", encoded_ordinal_data)

# If you want to inverse transform the encoded data back to original categories
decoded_data = label_encoder.inverse_transform(encoded_ordinal_data)
print("Decoded data:", decoded_data)
