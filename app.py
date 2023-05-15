import csv
import sqlalchemy as db

# Define the database connection string
db_uri = 'sqlite:///training_data.db'
# Create an engine object
engine = db.create_engine(db_uri)
connection = engine.connect()

# Create a metadata object
meta_data = db.MetaData()

# Create a table object
table_1 = db.Table('training_data', meta_data,
    db.Column('X', db.REAL),
    db.Column('Y1 (training func)', db.REAL),
    db.Column('Y2 (training func)', db.REAL),
    db.Column('Y3 (training func)', db.REAL),
    db.Column('Y4 (training func)', db.REAL)
)

# Create the table in the database
meta_data.create_all(engine)

# Open the CSV files
csv_files = ['training_data_1.csv', 'training_data_2.csv', 'training_data_3.csv', 'training_data_4.csv']

# Iterate over the CSV files
for csv_file in csv_files:
    # Open the CSV file
    with open(csv_file, 'r') as f:
        # Read the CSV file
        reader = csv.reader(f)
        # Skip the header row
        next(reader)
        # Iterate over the rows in the CSV file
        for row in reader:
            # Convert the values to float
            row = [float(value) for value in row]
            # Extract individual column values
            x_value, y_value = row
            # Insert the row into the database
            engine.execute(table_1.insert().values(
                X=x_value,
                **{
                    'Y1 (training func)': y_value,
                    'Y2 (training func)': y_value,
                    'Y3 (training func)': y_value,
                    'Y4 (training func)': y_value
                }
            ))

# Create a connection to the database
with engine.connect() as conn:
    # Execute a query to select all rows from the table
    result = conn.execute(table_1.select())
    # Fetch all rows from the result
    rows = result.fetchall()
    # Print the table
    print('| X | Y1 (training func) | Y2 (training func) | Y3 (training func) | Y4 (training func) |')
    print('|---|-------------------|-------------------|-------------------|-------------------|')
    for row in rows:
        print('| {} | {} | {} | {} | {} |'.format(row['X'], row['Y1 (training func)'], row['Y2 (training func)'], row['Y3 (training func)'], row['Y4 (training func)']))

# Close the database connection
engine.dispose()
