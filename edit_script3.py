import nbformat

# Load the notebook
notebook_path = r'c:\Users\walimunige.rupasingh\OneDrive - University of Calgary\Documents\GitHub\snowdroughtindex-main\notebooks\workflows\0.2_gapfilling_data_preparation_workflow.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# The cell is the 10th code cell, but let's find it by content
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'Extract SWE data' in cell.source:
        print(f"Found at index {i}")
        print("Old source:", cell.source)
        # New source
        new_source = [
            "# Extract SWE data for the selected stations",
            "selected_station_ids = stations_in_basin['station_id'].values",
            "bow_canswe_selected = bow_canswe",
            "# Convert to DataFrame",
            "bow_canswe_selected_df = bow_canswe_selected.reset_index()",
            "display(bow_canswe_selected_df.head())"
        ]
        cell.source = new_source
        print("New source:", cell.source)
        break

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated")