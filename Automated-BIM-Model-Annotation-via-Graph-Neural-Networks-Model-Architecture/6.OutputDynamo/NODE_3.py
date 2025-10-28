# Load the Python Standard and DesignScript Libraries
import sys
import clr
clr.AddReference('ProtoGeometry')
from Autodesk.DesignScript.Geometry import *

# The inputs to this node will be stored as a list in the IN variables.
dataEnteringNode = IN

# Get CSV data from Import CSV node
csv_data = IN[0]

print("=== CSV PROCESSING ===")
print(f"CSV data type: {type(csv_data)}")
print(f"CSV data length: {len(csv_data) if csv_data else 'None'}")

# Debug: Show the actual CSV content
if csv_data:
    print(f"First few rows of CSV: {csv_data[:5]}")
    print(f"CSV structure details:")
    for i, row in enumerate(csv_data[:5]):
        print(f"  Row {i}: {row} (type: {type(row)}, length: {len(row) if row else 'None'})")
else:
    print("CSV data is None or empty!")

# Initialize data containers
predicted_labels = []
original_labels = []

if csv_data and len(csv_data) >= 2:  # Need at least 2 rows (header + data)
    print(f"CSV structure: {csv_data[:3]}")  # Debug: show first 3 rows
    
    # CSV data is now properly formatted (thanks to Transpose = True in Import CSV)
    # First row is header: ['label_type', 'predicted_label']
    # Subsequent rows are data: ['0', '0'], ['1', '0'], etc.
    
    # Skip the header row (index 0) and process data rows
    original_labels = []
    predicted_labels = []
    
    print(f"Processing {len(csv_data)-1} data rows (skipping header)")
    
    for i, row in enumerate(csv_data[1:]):  # Skip header row
        print(f"Processing row {i+1}: {row}")
        if len(row) >= 2:  # Ensure row has both columns
            try:
                original_labels.append(int(row[0]))  # label_type column
                predicted_labels.append(int(row[1]))  # predicted_label column
                print(f"  Successfully added: original={row[0]}, predicted={row[1]}")
            except ValueError as e:
                print(f"  Error converting row {row}: {e}")
                continue
        else:
            print(f"  Row {i+1} has insufficient columns: {len(row)}")
    
    print(f"Final counts: {len(original_labels)} original labels, {len(predicted_labels)} predicted labels")
    
    # Create dummy data for elements (since we only have predictions)
    # In a real workflow, you'd combine this with your friend's element data
    num_elements = len(predicted_labels)
    unique_ids = [f"dummy_element_{i}" for i in range(num_elements)]
    element_types = ["Unknown"] * num_elements
    coordinates = [(0.0, 0.0, 0.0)] * num_elements  # Default coordinates
    annotation_texts = []
    info_strings = []
    
    # Generate annotation texts based on predicted labels
    for label in predicted_labels:
        if label == 0:
            annotation_texts.append("No Label")
        elif label == 1:
            annotation_texts.append("Wall Dimension")
        elif label == 2:
            annotation_texts.append("Connected Label")
        elif label == 3:
            annotation_texts.append("Door Marker")
        elif label == 4:
            annotation_texts.append("Zone Stamp")
        else:
            annotation_texts.append("Unknown")
    
    # Generate info strings
    info_strings = [f"Prediction for element {i}" for i in range(num_elements)]

print(f"Extracted {len(predicted_labels)} predictions")
print(f"Original labels: {original_labels[:5]}")
print(f"Predicted labels: {predicted_labels[:5]}")

# Assign your output to the OUT variable.
OUT = unique_ids, element_types, coordinates, predicted_labels, annotation_texts, info_strings