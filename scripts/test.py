from pathlib import Path
import json

p = list(Path(r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final\labels").glob("*.json"))[0]

print("sample:", p)

data = json.load(open(p))

print("keys:", data.keys())
print(data)