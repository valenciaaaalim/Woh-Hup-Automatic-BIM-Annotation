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

from RevitServices.Persistence import DocumentManager
from RevitServices.Transactions import TransactionManager
from Autodesk.Revit.DB import *

doc = DocumentManager.Instance.CurrentDBDocument

def create_text_annotation(point, text):
    """Create a text annotation at the specified point"""
    try:
        view = doc.ActiveView
        
        # Get existing text note types from the document
        collector = FilteredElementCollector(doc).OfClass(TextNoteType)
        text_note_types = list(collector)
        
        if not text_note_types:
            print("No text note types found in document")
            return None
        
        # Use the first available text note type
        text_note_type = text_note_types[0]
        print(f"Using text note type: {text_note_type.Name}")
        
        # Create the text note (updated method signature for newer Revit versions)
        text_note = TextNote.Create(doc, view.Id, point, text, text_note_type.Id)
        print(f"Successfully created text annotation: {text} at {point}")
        return text_note
            
    except Exception as e:
        print(f"Error creating text annotation: {e}")
        return None

def create_door_marker(point):
    """Create a door marker annotation"""
    try:
        view = doc.ActiveView
        
        # Get existing text note types from the document
        collector = FilteredElementCollector(doc).OfClass(TextNoteType)
        text_note_types = list(collector)
        
        if not text_note_types:
            print("No text note types found for door marker")
            return None
        
        # Use the first available text note type
        text_note_type = text_note_types[0]
        
        # Create the door marker (updated method signature for newer Revit versions)
        text_note = TextNote.Create(doc, view.Id, point, "🚪", text_note_type.Id)
        print(f"Successfully created door marker at {point}")
        return text_note
            
    except Exception as e:
        print(f"Error creating door marker: {e}")
        return None

def create_zone_stamp(point, text):
    """Create a zone stamp annotation"""
    try:
        view = doc.ActiveView
        
        # Get existing text note types from the document
        collector = FilteredElementCollector(doc).OfClass(TextNoteType)
        text_note_types = list(collector)
        
        if not text_note_types:
            print("No text note types found for zone stamp")
            return None
        
        # Use the first available text note type
        text_note_type = text_note_types[0]
        
        # Create the zone stamp (updated method signature for newer Revit versions)
        text_note = TextNote.Create(doc, view.Id, point, f"ZONE: {text}", text_note_type.Id)
        print(f"Successfully created zone stamp: ZONE: {text} at {point}")
        return text_note
            
    except Exception as e:
        print(f"Error creating zone stamp: {e}")
        return None

def process_annotations(elements, element_types, coordinates, predicted_labels, annotation_texts, info_strings):
    """Process all elements and create appropriate annotations"""
    
    print(f"=== PROCESSING ANNOTATIONS ===")
    print(f"Elements to process: {len(elements)}")
    print(f"Predicted labels: {predicted_labels}")
    print(f"Active view: {doc.ActiveView.Name}")
    
    TransactionManager.Instance.EnsureInTransaction(doc)
    
    created_annotations = []
    errors = []
    
    try:
        for i, (element, elem_type, coords, label, text, info) in enumerate(zip(
            elements, element_types, coordinates, predicted_labels, annotation_texts, info_strings)):
            
            print(f"\n--- Processing element {i} ---")
            print(f"Element type: {elem_type}")
            print(f"Predicted label: {label}")
            print(f"Annotation text: {text}")
            print(f"Coordinates: {coords}")
            
            if element is None:
                errors.append(f"Element {i}: Not found in Revit model")
                print(f"Element {i}: Skipped - element is None")
                continue
            
            # Use coordinates directly for annotation placement
            if isinstance(element, dict) and 'coordinates' in element:
                point_coords = element['coordinates']
            else:
                point_coords = coords
            
            # Convert coordinates to Revit XYZ
            point = XYZ(point_coords[0], point_coords[1], point_coords[2])
            print(f"Creating annotation at point: {point}")
            
            # Create annotation based on predicted label
            annotation = None
            
            if label == 0:  # No Label
                print(f"Label 0: No annotation needed")
                pass  # Do nothing
                
            elif label == 1:  # Wall with Dimension
                print(f"Label 1: Creating wall dimension annotation")
                annotation = create_text_annotation(point, "DIM")
                
            elif label == 2:  # Connected Label
                print(f"Label 2: Creating connected label annotation")
                annotation = create_text_annotation(point, text)
                
            elif label == 3:  # Door Marker
                print(f"Label 3: Creating door marker annotation")
                annotation = create_door_marker(point)
                
            elif label == 4:  # Zone Stamp
                print(f"Label 4: Creating zone stamp annotation")
                annotation = create_zone_stamp(point, text)
            
            if annotation:
                created_annotations.append(annotation)
                print(f"✅ Successfully created annotation for {elem_type}: {text} at {point_coords}")
            else:
                error_msg = f"Failed to create annotation for {elem_type} (label: {label})"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
    
    except Exception as e:
        error_msg = f"Transaction error: {e}"
        errors.append(error_msg)
        print(f"❌ {error_msg}")
    finally:
        TransactionManager.Instance.TransactionTaskDone()
        print(f"Transaction completed")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Created {len(created_annotations)} annotations")
    print(f"Encountered {len(errors)} errors")
    
    return created_annotations, errors

# Get inputs from previous nodes
# IN[0] should be: (elements, element_types, coordinates, predicted_labels, annotation_texts, info_strings, element_errors)
print(f"=== NODE 5 DEBUG ===")
print(f"Input received: {type(IN[0])}")
print(f"Input length: {len(IN[0]) if IN[0] else 'None'}")

if IN[0] and len(IN[0]) == 7:
    elements = IN[0][0]
    element_types = IN[0][1]
    coordinates = IN[0][2]
    predicted_labels = IN[0][3]
    annotation_texts = IN[0][4]
    info_strings = IN[0][5]
    element_errors = IN[0][6]
    
    print(f"Elements count: {len(elements)}")
    print(f"Predicted labels: {predicted_labels[:10]}")
    print(f"Non-zero labels: {[label for label in predicted_labels if label != 0]}")
    print(f"Annotation texts: {annotation_texts[:5]}")
    
    # Process annotations
    created_annotations, annotation_errors = process_annotations(
        elements, element_types, coordinates, predicted_labels, annotation_texts, info_strings
    )
    
    # Combine all errors
    all_errors = element_errors + annotation_errors
    
    # Assign your output to the OUT variable.
    OUT = created_annotations, all_errors, f"Created {len(created_annotations)} annotations"
else:
    print(f"Error: Expected 7 inputs from previous node, got {len(IN[0]) if IN[0] else '0'}")
    OUT = [], ["Insufficient inputs"], "Failed to process"