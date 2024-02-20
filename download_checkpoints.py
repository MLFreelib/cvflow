import os
import gdown

# Change directory to examples
os.chdir('examples')

# Download from Google Drive
gdown.download('https://drive.google.com/drive/folders/1YPBDpGG3spgh7J8HKUtmDJi7KX7BwNmJ?usp=sharing', output=None, quiet=False)

# Move downloaded file to checkpoints directory
os.rename('Веса', 'checkpoints')