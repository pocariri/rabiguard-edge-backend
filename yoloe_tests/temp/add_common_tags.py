import os

common_tags = [
    "air conditioner",
    "cabinet",
    "closet",
    "door",
    "window",
    "television",
    "monitor",
    "shelf",
    "heater"
]

files_to_update = [
    "retail_and_convenience_store.txt",
    "food_and_beverage.txt",
    "smart_home_and_elderly_care.txt",
    "industrial_and_logistics.txt",
    "education_and_daycare.txt"
]

for filename in files_to_update:
    filepath = os.path.join("yoloe_tests", filename)
    
    # Read existing tags
    with open(filepath, "r") as f:
        existing_tags = set(line.strip() for line in f if line.strip())
        
    # Combine and sort (to keep them neat)
    all_tags = sorted(list(existing_tags.union(set(common_tags))))
    
    # Write back
    with open(filepath, "w") as f:
        for tag in all_tags:
            f.write(tag + "\n")
            
    print(f"Updated {filename} with common tags. Total tags: {len(all_tags)}")

