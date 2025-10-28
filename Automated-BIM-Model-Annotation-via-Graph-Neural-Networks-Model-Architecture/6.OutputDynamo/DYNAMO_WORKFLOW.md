# Complete Dynamo Workflow Guide: CSV to Annotated Floor Plans

## 🎯 Overview

This guide provides a complete Dynamo workflow to process CSV output from machine learning models and create annotated floor plans in Revit. The pipeline reads CSV predictions and automatically places annotations on BIM elements, transforming machine learning results into visual BIM annotations.

## 📊 Data Flow

```
CSV File (ML Model Output) → Dynamo Script → Revit Annotations
```

**Input:** CSV file with element predictions from machine learning model  
**Output:** Annotated floor plans in Revit with visual annotations

## 🚀 Quick Start

### Step 1: Install Required Packages
1. Open Dynamo
2. Go to `Packages` → `Search for a Package`
3. Install these packages:
   - **None required!** (All nodes are built into Dynamo core)

### Step 2: Create Your Dynamo Script

**Node Sequence:**
```
File Path → Import CSV → Python Script (Data Processing) → Python Script (Element Retrieval) → Python Script (Annotation Creation) → Watch
```

## 🔧 Detailed Node Configuration

### Node 1: File Path
- **Node Type:** `File Path`
- **Input:** Set path to CSV file containing ML model predictions
- **Example:** `C:\path\to\predictions_output_graph_10.csv`

### Node 2: Import CSV
- **Node Type:** `Import CSV`
- **Input:** Connect File Path output
- **Configuration:** No additional settings needed

### Node 3: Python Script (Data Processing)

**Purpose:** Parse CSV data and extract columns

**Input:** CSV data from Import CSV node  
**Output:** Structured data for processing

> Copy paste from NODE_3.py file

### Node 4: Python Script (Element Retrieval)

**Purpose:** Find Revit elements by GUID or create coordinate-based annotations

**Input:** Structured data from previous script  
**Output:** Element references and coordinates

> Copy paste from NODE_4.py file

### Node 5: Python Script (Annotation Creation)

**Purpose:** Create actual annotations in Revit based on predicted labels

**Input:** Element data and predictions  
**Output:** Created annotations and status

> Copy paste from NODE_5.py file

### Node 6: Watch (Optional)
- **Node Type:** `Watch`
- **Input:** Connect output from Annotation Creation script
- **Purpose:** Monitor results and debug

## 📋 CSV File Format Requirements

Your model's output CSV file has this format:

| Column | Description | Example |
|--------|-------------|---------|
| `label_type` | Original annotation type | `0`, `1`, `2`, `3`, `4` |
| `predicted_label` | Model's prediction | `0`, `1`, `2`, `3`, `4` |

**Note:** This workflow processes ML model predictions from CSV files. The script generates dummy element data and coordinates for demonstration purposes. In a production workflow, this would be combined with actual BIM element data to obtain real coordinates and element information.

## 🏷️ Annotation Types

| Label | Type | What It Creates |
|-------|------|-----------------|
| **0** | No Label | Does nothing (removes existing annotations) |
| **1** | Wall with Dimension | Creates "DIM" text annotation |
| **2** | Connected Label | Creates custom text annotation |
| **3** | Door Marker | Creates 🚪 symbol |
| **4** | Zone Stamp | Creates "ZONE: [text]" annotation |

## 🚨 Troubleshooting

### Common Issues and Solutions

**1. Elements Not Found**
- **Problem:** GUIDs in CSV don't match Revit elements
- **Solution:** Verify GUID format and element existence in model

**2. Python Script Errors**
- **Problem:** Syntax errors or missing references
- **Solution:** Check Python syntax and ensure all required references are loaded

**3. Annotations Not Created**
- **Problem:** Revit permissions or view issues
- **Solution:** Check Revit permissions and ensure active view is appropriate

**4. Package Not Found**
- **Problem:** Required Dynamo packages missing
- **Solution:** All required nodes are built into Dynamo core - no additional packages needed

**5. CSV Data Issues**
- **Problem:** Incorrect CSV format or missing columns
- **Solution:** Verify CSV has `label_type` and `predicted_label` columns from ML model output

## 📊 Expected Results

### Successful Execution Output
When the script runs successfully, the console will display:

- ✅ **Successful Annotations:** "Successfully created door marker at (coordinates)"
- ✅ **Zone Stamps:** "Successfully created zone stamp: ZONE: [text] at (coordinates)"
- ✅ **Final Count:** "Created X annotations"
- ✅ **Transaction Status:** "Transaction completed"

### Error Labels (Expected Behavior)
The following "errors" are actually normal behavior for label 0 predictions:

- ❌ **Failed to create annotation for Unknown (label: 0)** - This indicates "No Label" predictions, which correctly skip annotation creation
- ❌ **Encountered X errors** - These are label 0 cases that don't require annotations

### What You'll See in Revit
In the active Revit floor plan view, you will observe:

- **🚪 Door Markers:** Multiple door symbols (🚪) placed at predicted door locations
- **Zone Stamps:** Text annotations displaying "ZONE: Zone Stamp" for predicted zone areas
- **Annotation Placement:** All annotations positioned at coordinates (0,0,0) - the origin point
- **Text Note Elements:** All annotations appear as Revit TextNote elements in the model

## 🎯 Next Steps

1. **Test with ML Model Data:** Use CSV output from machine learning models
2. **Customize Annotations:** Modify the Python scripts for specific annotation requirements
3. **Add More Annotation Types:** Extend the script for additional annotation categories
4. **Optimize Performance:** Implement batch processing for large datasets
5. **Create User Interface:** Add input parameters for file paths and configuration settings

## 📁 Required Files

- **CSV File:** ML model predictions containing `label_type` and `predicted_label` columns
- **Dynamo Script:** The workflow described above
- **Revit Model:** The BIM model to annotate

## 🔄 Complete Workflow Summary

1. **Input:** CSV file with element predictions from machine learning model
2. **Process:** Dynamo script reads CSV and extracts prediction data
3. **Prepare:** Generate element data and coordinate information
4. **Annotate:** Create appropriate annotations based on ML predictions using Revit API
5. **Output:** Annotated floor plans in Revit with visual BIM elements

This workflow provides a complete pipeline from machine learning predictions to annotated BIM models, enabling automated annotation of floor plans based on AI model outputs.
