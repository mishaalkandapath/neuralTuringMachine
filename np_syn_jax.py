import time
# for accidental numpy mutation while coding in jax:
with open("transformer.py", "r") as f:
    lines = f.readlines()
    for line_idx, line in enumerate(lines):
        #the assumption is every line has only one command
        # every matrix is jax ndarray
        # for the future, track variable types
        for idx, character in enumerate(line):
            if character == "[" \
            and "=" not in line[:idx] \
            and ":" in line[idx+1:line[idx+1:].index("]")] \
            and "=" in line[line[idx+1:].index("]")+1:] \
            and "==" not in line[line[idx+1:].index("]")+1:]:
                end_bracks = line[idx+1:].index("]")
                old_line = line
                line = old_line[:idx] + ".at" + old_line[idx: len(line[:idx+1])+end_bracks+1]+".set("
                #check for comments beginning)
                if "#" in line[end_bracks + 1:]:
                    comment_idx = list(old_line[len(line[:idx+1])+end_bracks + 1:]).index("#")
                    equals_idx = old_line[len(line[:idx+1])+end_bracks + 1:].index("=")
                    more_offset = len(old_line[:len(line[:idx+1])+end_bracks + 1])
                    line += old_line[more_offset + equals_idx + 1: more_offset + comment_idx] + ")\n"
                    lines[line_idx] = line  
                else:
                    equals_idx = old_line[len(line[:idx+1])+end_bracks + 1:].index('=')
                    more_offset = len(old_line[:len(line[:idx+1])+end_bracks + 1])
                    line += old_line[more_offset + equals_idx + 1:-1] + ")\n"
                    lines[line_idx] = line
            else:
                continue
with open("transformer.py", "w") as f:
    f.writelines(lines)