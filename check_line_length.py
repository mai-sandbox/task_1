#!/usr/bin/env python3
"""Script to check for lines longer than 100 characters in agent.py"""

with open('/home/daytona/task_1/agent.py', 'r') as f:
    lines = f.readlines()

long_lines = []
for i, line in enumerate(lines, 1):
    if len(line.rstrip()) > 100:
        long_lines.append((i, len(line.rstrip()), line.rstrip()))

if long_lines:
    print(f"Found {len(long_lines)} lines longer than 100 characters:")
    for line_num, length, content in long_lines:
        print(f"Line {line_num} ({length} chars): {content}")
else:
    print("No lines longer than 100 characters found.")
