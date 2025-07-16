"""
This file control the transformation of the stage code from original combined data to the dataset for the classification task.
"""


# stage_code = {
#     '0': 1, # wake
#     '1': -1, # REM
#     '2': 0, # N1
#     '3': -1, # N2
#     '4': -1, # N3
#     '5': -1, # N4
# }

stage_code = {
    '11': 1, # wake
    '12': -1, # REM
    '13': 0, # N1
    '14': -1, # N2
    '15': -1, # N3
    '16': -1, # N4
}

stage_code_global = {
    '11': 0, # wake
    '12': 1, # REM
    '13': 2, # N1
    '14': 3, # N2
    '15': 4, # N3
    '16': 4, # N4
}

def _code_to_str(code_dict):
    # Convert stage code dictionary to a string representation
    return ''.join('n' if code_dict[key] == -1 else str(code_dict[key]) for key in sorted(code_dict.keys()))

def info(code_dict):
    # Print the stage code information
    print("Stage Code Info:")
    for key, value in code_dict.items():
        print(f"Stage {key}: {value}")

stage_code_str = _code_to_str(stage_code)
stage_code_global_str = _code_to_str(stage_code_global)