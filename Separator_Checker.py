with open("D:/AI_Sample_06152023/Adjoin/Result.txt", 'r') as file:
    first_line = file.readline()  # Read the first line
    separator = ','
    if '\t' in first_line:
        separator = '\\t'
    elif ';' in first_line:
        separator = ';'
    # Add more conditions for other possible separators if needed

print(f"The separator used in the file is: {separator}")