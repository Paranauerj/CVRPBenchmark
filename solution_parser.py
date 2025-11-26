def parse_solution_file(file_obj):
    """Parses CVRPLIB .sol file for Cost."""
    bks_cost = None
    for line in file_obj:
        if hasattr(line, 'decode'): line = line.decode('utf-8')
        line = line.strip().lower()
        if line.startswith("cost"):
            try:
                bks_cost = int(line.split()[-1])
                break
            except: pass
    if bks_cost is None: raise ValueError("No Cost found in .sol file")
    return bks_cost