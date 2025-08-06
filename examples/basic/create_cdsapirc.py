import os

# Get the home directory
home_dir = os.path.expanduser("~")

# Create the .cdsapirc file path
cdsapirc_path = os.path.join(home_dir, ".cdsapirc")

# Write the configuration with the correct URL
with open(cdsapirc_path, "w") as f:
    f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
    f.write("key: 36067d72-62ff-45fe-b107-c3d871496230\n")

print(f"Created .cdsapirc file at: {cdsapirc_path}") 