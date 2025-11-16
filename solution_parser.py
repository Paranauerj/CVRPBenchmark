def parse_solution_file(file_obj):
    """
    Parses a CVRPLIB .sol file from a file-like object.
    
    The file_obj can be from open() or a Streamlit FileUploader.
    
    Returns a tuple: (bks_cost as int, bks_routes as list of strings)
    """
    bks_cost = None
    bks_routes = []
    
    for line in file_obj:
        # Handle both binary (from upload) and text (from open)
        if hasattr(line, 'decode'):
            line = line.decode('utf-8')
        
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        if line_lower.startswith("cost"):
            try:
                # Find the number on the line, e.g., "Cost 63684"
                bks_cost = int(line.split()[-1])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse cost from line: {line}")
        elif line_lower.startswith("route"):
            bks_routes.append(line_stripped)
                
    if bks_cost is None:
        raise ValueError("Could not find 'Cost' line in solution file.")
        
    return bks_cost, bks_routes