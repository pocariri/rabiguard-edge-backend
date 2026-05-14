import os

# These are objects that are either too small, highly movable, or not suitable to act as a FIXED monitoring zone (ROI).
small_or_movable_tags = [
    "cart",          # small or movable
    "shopping cart", # movable
    "chair",         # movable
    "box",           # movable/variable size
    "crate",         # movable/variable size
    "barrel",        # movable/variable size
    "monitor",       # too small for a general ROI zone (unless specific close-up)
    "television",    # border-line, but usually people want to monitor the AREA, not just the TV itself. Let's remove to be safe based on "small/movable".
    "coffee machine",# too small
    "microwave",     # too small
    "oven"           # border-line, but often small or built-in.
]

# Let's keep: door, window, counter, shelf, cabinet, closet, bed, couch, armchair, stairs, sink, table, air conditioner, heater, atm, forklift, truck, machine, container, fence, blackboard, whiteboard, playground, school bus.

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
        existing_tags = [line.strip() for line in f if line.strip()]
        
    # Filter out small/movable tags
    filtered_tags = [tag for tag in existing_tags if tag not in small_or_movable_tags]
    
    # Write back
    with open(filepath, "w") as f:
        for tag in filtered_tags:
            f.write(tag + "\n")
            
    print(f"Updated {filename}: removed {len(existing_tags) - len(filtered_tags)} tags. Remaining: {len(filtered_tags)}")

