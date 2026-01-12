import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumendraprasad/musical-instruments-sound-dataset")

print("Path to dataset files:", path)