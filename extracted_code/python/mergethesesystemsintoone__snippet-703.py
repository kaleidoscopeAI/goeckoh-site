"""
Loads data from a CSV, performs some basic preprocessing, and saves it to a new CSV.
"""
try:
    df = pd.read_csv(input_file)

    # Example preprocessing steps:
    df.dropna(inplace=True)  # Remove rows with missing values
    df['column1'] = df['column1'].str.lower()  # Convert a column to lowercase

    df.to_csv(output_file, index=False)
    print(f"Data preprocessed and saved to {output_file}")

except Exception as e:
    print(f"Error during preprocessing: {e}")
    sys.exit(1)  # Exit with an error code

if len(sys.argv) != 3:
    print("Usage: python preprocess_data.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
preprocess_data(input_file, output_file)






