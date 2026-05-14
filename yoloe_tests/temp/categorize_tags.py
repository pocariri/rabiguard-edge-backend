import os

env_tags = {
    "retail_and_convenience_store.txt": [
        "cash register", "refrigerator", "door", "shelf", "counter", 
        "basket", "shopping cart", "display window", "convenience store", 
        "atm", "cart", "bottle", "can", "bag"
    ],
    "food_and_beverage.txt": [
        "counter", "oven", "microwave", "dining table", "table", 
        "chair", "cup", "bottle", "coffee machine", "refrigerator", 
        "tray", "menu", "plate", "bowl", "fork", "knife", "spoon", 
        "coffee cup", "wine glass", "sink"
    ],
    "smart_home_and_elderly_care.txt": [
        "couch", "bed", "door", "table", "medicine", "pharmacy", 
        "dog", "cat", "window", "tv", "armchair", "bathtub", "shower", 
        "stairs", "refrigerator", "toilet", "sofa"
    ],
    "industrial_and_logistics.txt": [
        "conveyor", "forklift", "fence", "box", "crate", "truck", 
        "helmet", "factory", "warehouse", "machine", "pallet", 
        "container", "barrel", "hard hat"
    ],
    "education_and_daycare.txt": [
        "desk", "stairs", "chair", "water dispenser", "book", 
        "blackboard", "whiteboard", "toy", "playground", "backpack", 
        "school bus", "pencil", "pen", "notebook"
    ]
}

# Read ram_tag_list.txt
with open("yoloe_tests/ram_tag_list.txt", "r") as f:
    valid_tags = set(line.strip().lower() for line in f)

# Filter and write to files
for filename, tags in env_tags.items():
    filtered_tags = [tag for tag in tags if tag.lower() in valid_tags]
    filepath = os.path.join("yoloe_tests", filename)
    with open(filepath, "w") as f:
        for tag in filtered_tags:
            f.write(tag + "\n")
    print(f"Created {filename} with {len(filtered_tags)} tags.")

