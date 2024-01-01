import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('../result/output_data.csv')

# Calculate mean for each column
column_means = data.mean()

# Calculate the errors for each column
errors = {}
for col in data.columns:
    print(col)
    errors[col] = data[col] - column_means[col]

# Convert errors to DataFrame
errors_df = pd.DataFrame(errors)

abs_errors = errors_df.abs()
# Calculate median errors for each column
median_errors = abs_errors.median()

# Plotting mean errors for each column
plt.figure(figsize=(8, 6))

for col in errors_df.columns:
    plt.plot(errors_df.index, errors_df[col], label=col)

color_list = ['blue', 'red', 'green']
count = 0
for col, median_error in median_errors.items():
    plt.axhline(y=median_error, color=color_list[count], linestyle='--', label=f'{col} Median Error')
    count += 1
plt.title('Mean Er rors for Columns')
plt.xlabel('Rows')
plt.ylabel('Mean Error')
plt.legend()
plt.grid(True)
plt.show()
