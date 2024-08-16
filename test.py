import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = '/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/combinedOCV.csv'
df = pd.read_csv(csv_file_path)

# Save the DataFrame to an Excel file
excel_file_path = '/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/combinedOCV.xlsx'
df.to_excel(excel_file_path, index=False)

