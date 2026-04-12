#!/bin/bash

# OpenCAS Automated Audit Workflow
# This script triggers the Gemini CLI to analyze the OpenCAS repository and
# generate two markdown reports (Overall State & Delta State).

AUDIT_DIR="(workspace_root)/docs/audits"
mkdir -p "$AUDIT_DIR"

DATE=$(date +%Y-%m-%d_%H-%M)
LATEST_OVERALL=$(ls -t "$AUDIT_DIR"/Overall_State_*.md 2>/dev/null | head -1)

if [ -z "$LATEST_OVERALL" ]; then
    PREV_MSG="No previous overall state report found. Compare directly against the baseline docs."
else
    PREV_MSG="The most recent overall state report is located at: $LATEST_OVERALL"
fi

echo "=========================================================="
echo " Starting OpenCAS Progress Audit Workflow"
echo " Date: $DATE"
echo "=========================================================="
echo ""
echo "Triggering Gemini CLI to generate reports..."

# The prompt instructions for the Gemini CLI
PROMPT="Run the OpenCAS Progress Audit workflow:
1. Scan the current state of the (workspace_root) repository.
2. Read the initial baseline report: (workspace_root)/docs/opencas-architecture-and-comparison.md.
3. Read the previous state: $PREV_MSG.
4. Generate and write a file to '$AUDIT_DIR/Overall_State_$DATE.md' that compares the current state of the repo directly to the initial baseline report. Include metrics, fixed gaps, and remaining gaps. Use the 'run_shell_command' to create the file if it falls outside of your current workspace boundary.
5. Generate and write a file to '$AUDIT_DIR/Delta_State_$DATE.md' that outlines ONLY the changes made since the previous overall state report (or since the baseline if no previous report exists). Use the 'run_shell_command' to create the file if it falls outside of your current workspace boundary.
6. Do not ask for confirmation, just write the two files and provide a brief summary in the chat."

# Execute Gemini CLI with the prompt
gemini -p "$PROMPT"

echo ""
echo "=========================================================="
echo " Audit Workflow Complete!"
echo " Reports should be available in: $AUDIT_DIR/"
echo "=========================================================="
