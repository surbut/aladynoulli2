#!/bin/bash
# Using screen (alternative to nohup)

# Start a new screen session
screen -S mediation_analysis

# Then inside the screen session, run:
# cd /path/to/your/script/directory
# Rscript mediation_analysis.R
#
# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r mediation_analysis
# List sessions: screen -ls

