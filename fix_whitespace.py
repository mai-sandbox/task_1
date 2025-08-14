#!/usr/bin/env python3
"""Script to fix whitespace issues in agent.py"""

import re

# Read the file
with open('/home/daytona/task_1/agent.py', 'r') as f:
    content = f.read()

# Remove trailing whitespace from each line
content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

# Remove blank lines at the end of the file
content = content.rstrip()

# Ensure file ends with exactly one newline
content += '\n'

# Write back to file
with open('/home/daytona/task_1/agent.py', 'w') as f:
    f.write(content)

print("Fixed whitespace issues in agent.py")
