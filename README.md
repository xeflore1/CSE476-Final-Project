# CSE476-Final-Project

This repository contains our implementation of an agent that uses 10 prompting techniques (split evenly amongst 5 members) to solve problems from math, coding, future prediction, common sense, and planning
Prompting Techniques: Chain of Thought, Self-Consistency, Self-Refine, Tree of Thought, ReACT, Tool-Augmented Reasoning, LLM as Judge, Prompt Optimization, Decomposition, Ensemble Voting

#### Requirements
- Python 3.10 or newer (3.14 is what we develop on; both work).
- LLM API Key located in .env file
- Being Present in ASU WIFI network or using CISCO SSL VPN for remote connection

## Setup

bash setup:
```
# From repo root
cd CSE476-Final-Project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Create the env file (this file is git-ignored)
cat > .env <<'EOF'
API-KEY=<api-key>
API_BASE=https://openai.rc.asu.edu/v1
MODEL_NAME=qwen3-30b-a3b-instruct-2507
EOF
```
All commands assume you are inside `CSE476-Final-Project/` with the venv active.

## Full submission run (6,208 questions)

```bash
python3 run_submission.py
```

This file acts as the "main" file for our program. It inputs the questions from the cse_476_final_project_test_data.json, runs agent_router on all of our techniques, and outputs the final answer out to an cse_476_final_project_answers.json file

