from sklearn.preprocessing import LabelEncoder

# Sample categorical data
categories = ['red', 'blue', 'green', 'yellow', 'red', 'blue', 'green']

# Creating a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the categorical data
encoded_data = label_encoder.fit_transform(categories)

# Displaying the original and encoded data
print("Original categories:", categories)
print("Encoded data:", encoded_data)

# If you want to inverse transform the encoded data back to original categories
decoded_data = label_encoder.inverse_transform(encoded_data)
print("Decoded data:", decoded_data)
