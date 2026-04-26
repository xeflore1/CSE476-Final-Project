# CSE476-Final-Project

This repository contains our implementation of an agent that uses 10 prompting techniques (split evenly amongst 5 members) to solve problems from math, coding, future prediction, common sense, and planning
Prompting Techniques: Chain of Thought, Self-Consistency, Self-Refine, Tree of Thought, ReACT, Tool-Augmented Reasoning, LLM as Judge, Prompt Optimization, Decomposition, Ensemble Voting

#### Requirements
- Python 3.10 or newer (3.14 is what we develop on; both work).
- LLM API Key located in .env file
- Being Present in ASU WIFI network or using CISCO SSL VPN for remote connection

## Setup

```bash
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
## Single Question Test
```bash
python3 agent_router.py "What is 2 + 2?"
```

This file acts as the "main" file for our program. It inputs the questions from the cse_476_final_project_test_data.json, runs agent_router on all of our techniques, and outputs the final answer out to an cse_476_final_project_answers.json file

## Architecture of Repo

Flow of Program:
Input Question -> DC["Input"] -> Correct Domain

Correct Domain -> agent_router.agent(domain) -> prompt_optimization

prompt_optimized_call["question", domain] -> Reworded Prompt

Reworded Prompt -> Primary Technique["question"] -> answer

confidence_check("answer") -> If low:
<pre>
                                   ensemble_vote("question") OR self_refine(question) - depending on budget remaining
</pre>
```                                       
                              If medium OR high:
                                   submit answer and check for <= 5000 chars
```
If Tools Needed:
```
                  Primary Technique -> Uses /tools/calculator or tools/code_executor
```
## Primary Technique Mapping
'math': 'chain_of_thought'

'coding': 'tool_augmented'

'common_sense': 'chain_of_thought'

'planning': 'tree_of_thought'

'future_prediction': 'self_refine'

## File Structure:
/techniques
  chain_of_thought.py
  self_consistency.py
  tree_of_thought.py
  self_refine.py
  react_agent.py
  tool_augmented.py
  decomposition.py
  ensemble_voting.py
  prompt_optimization.py
  llm_as_judge.py
  output_instructions.py
  utils.py
/tools
  \_\_init\_\_.py
  calculator.py
  code_executor.py
  
.env
.gitignore
requirements.txt
api_wrapper.py
agent_router.py
run_submission.py    - Run this for submission
domain_classifier.py

cse476_final_project_dev_data.json    - 1,000 labelled dev rows
cse_476_final_project_test_data.json  - 6,208 unlabelled test rows
cse_476_final_project_answers.json    - generated submission


### Final File Format

The answers.json should look something like this:

[
  {
    "output": "ans1"
  },

  {
    "output": "ans2"
  }
]

## Important Considerations for Project

- Each Question has a budget of 20 calls
- Domain Classifier uses regex expression rather than api call to find domain
- Same method header for all techniques and return dictionary format: {'ok': Bool, 'text', 'answer', 'calls', 'error'}
- All methods call same call_model_chat_completions function for api calls
