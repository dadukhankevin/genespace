import os

with open('codebase.txt', 'w') as outfile:
    for file in os.listdir():
        if file.endswith('.py'):
            outfile.write(f"filename: {file}\n")
            with open(file, 'r') as infile:
                outfile.write(infile.read())
            outfile.write('\n')