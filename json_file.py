import json

# write python variable to json flile
def write_python_to_json_file(variable, path):
    '''
    variable: python variable
    path: json file path
    '''
    with open(path, 'w') as f:
        json.dump(vars(variable), f, indent=4)