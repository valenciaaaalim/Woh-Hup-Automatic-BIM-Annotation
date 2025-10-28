# Load the Python Standard and DesignScript Libraries
import sys
import clr
clr.AddReference('ProtoGeometry')
from Autodesk.DesignScript.Geometry import *

# The inputs to this node will be stored as a list in the IN variables.
dataEnteringNode = IN

# Place your code below this line
import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitServices')
clr.AddReference('System')

from RevitServices.Persistence import DocumentManager
from Autodesk.Revit.DB import *
import System

doc = DocumentManager.Instance.CurrentDBDocument

def create_dummy_elements(unique_ids, coordinates):
    """Create dummy elements for annotation purposes"""
    elements = []
    errors = []
    
    print(f"Creating {len(unique_ids)} dummy elements for annotation")
    
    for i, (guid_string, coords) in enumerate(zip(unique_ids, coordinates)):
        try:
            # Create a dummy element object with the coordinates
            dummy_element = {
                'guid': guid_string,
                'coordinates': coords,
                'index': i
            }
            elements.append(dummy_element)
            print(f"Created dummy element {i}: {guid_string} at {coords}")
            
        except Exception as e:
            elements.append(None)
            errors.append(f"Error creating dummy element {guid_string}: {e}")
    
    return elements, errors

# Get data from previous script
# IN[0] should be: (unique_ids, element_types, coordinates, predicted_labels, annotation_texts, info_strings)
print(f"=== NODE 4 DEBUG ===")
print(f"Input received: {type(IN[0])}")
print(f"Input length: {len(IN[0]) if IN[0] else 'None'}")

if IN[0] and len(IN[0]) == 6:
    unique_ids = IN[0][0]
    element_types = IN[0][1]
    coordinates = IN[0][2]
    predicted_labels = IN[0][3]
    annotation_texts = IN[0][4]
    info_strings = IN[0][5]
    
    print(f"Processing {len(unique_ids)} unique IDs")
    print(f"Predicted labels: {predicted_labels[:10]}")  # Show first 10 labels
    print(f"Non-zero labels: {[label for label in predicted_labels if label != 0]}")
    print(f"Annotation texts: {annotation_texts[:5]}")  # Show first 5 texts
    
    # Create dummy elements for annotation placement
    elements, element_errors = create_dummy_elements(unique_ids, coordinates)
    
    # Pass through ALL data plus new element data
    OUT = elements, element_types, coordinates, predicted_labels, annotation_texts, info_strings, element_errors
else:
    print(f"Error: Expected 6 inputs from previous node, got {len(IN[0]) if IN[0] else '0'}")
    if IN[0]:
        print(f"Input structure: {[type(item).__name__ for item in IN[0]]}")
    OUT = [], [], [], [], [], [], ["Error: Invalid input for Element Retrieval"]