INPUT_NAME = 'spiral'
OUTPUT_NAME = 'test'

PREFIX = 'experiments/datasets/'
SUFFIX = '.csv'
LINE_NUMBER_LIMIT = 10000

input = open(PREFIX + INPUT_NAME + SUFFIX, 'r')
output = open(PREFIX + OUTPUT_NAME + SUFFIX, 'w')

for i, line in enumerate(input):
    if i > LINE_NUMBER_LIMIT:
        break
    line = line[(line.index(';') + 1):]
    line = line.replace(';', ',')
    output.write(line)

input.close()
