import json
import pyMRAW
import sys

filename = sys.argv[1]

md = pyMRAW.get_cih(filename)

# Convert to plain Python types
metadata = dict(md)

print(json.dumps(metadata))