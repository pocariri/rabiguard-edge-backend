import os

env_tags = {
    "retail_and_convenience_store.txt": [
        "door", "shelf", "counter", "shopping cart", 
        "display window", "convenience store", "atm", "cart"
    ],
    "food_and_beverage.txt": [
        "counter", "oven", "microwave", "table", "chair", 
        "coffee machine", "sink"
    ],
    "smart_home_and_elderly_care.txt": [
        "couch", "bed", "door", "table", "window", 
        "armchair", "shower", "stairs"
    ],
    "industrial_and_logistics.txt": [
        "forklift", "fence", "box", "crate", "truck", 
        "factory", "warehouse", "machine", "container", "barrel"
    ],
    "education_and_daycare.txt": [
        "stairs", "chair", "blackboard", "whiteboard", 
        "playground", "school bus"
    ]
}

for filename, tags in env_tags.items():
    filepath = os.path.join("yoloe_tests", filename)
    with open(filepath, "w") as f:
        for tag in tags:
            f.write(tag + "\n")
    print(f"Updated {filename} with {len(tags)} tags.")

