import os
import sys
from subprocess import check_output


def main(folder):
    tot_num_lines = 0
    for root, folders, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                output = check_output(["wc", "-l", filepath])
                output = output.decode('utf-8').strip()
                num_lines, _ = output.split(' ')
                num_lines = int(num_lines)
                tot_num_lines += num_lines
    return tot_num_lines


if __name__ == '__main__':
    if len(sys.argv) == 1:
        folder = '.'
    else:
        folder = sys.argv[1]
    tot_num_lines = main(folder)
    print(tot_num_lines)
