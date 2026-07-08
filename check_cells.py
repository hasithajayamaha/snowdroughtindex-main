import nbformat

# Load the notebook
notebook_path = r'c:\Users\walimunige.rupasingh\OneDrive - University of Calgary\Documents\GitHub\snowdroughtindex-main\notebooks\workflows\0.2_gapfilling_data_preparation_workflow.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"Cell {i+1}: {cell.source[:50]}...")

print("Done")