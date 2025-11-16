def parse_solution_file(file_obj):
    """
    Parses a CVRPLIB .sol file from a file-like object.
    
    The file_obj can be from open() or a Streamlit FileUploader.
    
    Returns the BKS cost value as an integer.
    """
    bks_cost = None
    # bks_routes removed
    
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
                break # Found what we need
            except (ValueError, IndexError):
                print(f"Warning: Could not parse cost from line: {line}")
        # "elif" for routes removed
                
    if bks_cost is None:
        raise ValueError("Could not find 'Cost' line in solution file.")
        
    return bks_cost # <--- UPDATED: Only return cost