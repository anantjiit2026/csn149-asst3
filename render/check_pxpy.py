def find_missing_combinations(file_path):
    # Read the file and extract px, py values
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the px, py values from the lines
    coordinates = set()
    for line in lines:
        parts = line.strip().split(', ')
        print(parts)
        px = int(parts[0].split(': ')[1])
        py = int(parts[1].split(': ')[1])
        coordinates.add((px, py))

    # Define the range of px and py values
    px_range = range(0, 1023)
    py_range = range(0, 1023)

    # Find missing combinations
    # missing_combinations = []
    for px in px_range:
        for py_value in py_range:
            if (px, py_value) not in coordinates:
                # missing_combinations.append((px, py_value))
                print(f"Missing combination: px={px}, py={py_value}")

    # return missing_combinations

# Example usage
file_path = 'result.txt'
missing_combinations = find_missing_combinations(file_path)
print("Missing combinations:", missing_combinations)