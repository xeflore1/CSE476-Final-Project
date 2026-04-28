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
```

## Enter API Key into .env File
Enter your API key into the API-KEY=... value in the .env file before running the submission.


All commands assume you are inside `CSE476-Final-Project/` with the venv active.

## Full submission run (6,208 questions)

```bash
python3 run_submission.py
```
## Single Question Test
```bash
python3 agent_router.py "What is 2 + 2?"
```

This file acts as the "main" file for our program. It inputs the questions from the cse_476_final_project_test_data.json, runs agent_router.agent() for each question, and writes the final answer out to 'cse_476_final_project_answers.json'

## Architecture of Repo

Flow of Program:



Input Question -> domain_classifier.classify_domain("Input") -> Correct Domain  
Correct Domain -> agent_router.agent(prompt)  

'agent_router' runs prompt optimization ONLY WHEN prompt is not very long such as for (planning, future prediction, etc.)
prompt_optimized_call["question", domain] -> Optimized_or_Original Prompt  

Optimized_or_Original -> Primary Technique[default="Chain_of_Thought"] -> answer  

If primary answer is empty -> rescue_path(Retry with Chain_of_thought, THEN 'self_refine' if budget allows)  
If answer is non-empty and budget allows -> llm_as_judge.confidence_check  

confidence_check("answer") -> If low: ensemble_vote("question"); IF NOT:  self_refine(question) - depending on budget remaining  
confidence_check("answer") -> If medium OR high: submit answer and check for <= 5000 answers  

If Tools Needed: Primary Technique -> Uses /tools/calculator or tools/code_executor



## IMPORTANT: Chain of Thought is The Default Method for All Domains Due to its Effectiveness. That is why you might see it more frequently than other techniques.

## However, Other Techniques will still be Run Throughout the Submission


# Primary Technique Mapping AFTER Ensemble
'math': \['decomposition', 'tool_augmented'\]  
'common_sense': \['self_refine', 'self_consistency'\]  
'coding': \['react', 'self_refine'\]  
'future_prediction': \['self_consistency', 'self_consistency'\]  
'planning': \['decomposition', 'tree_of_thought'\]  

Notes:
- These are the per-domain ensemble mappings used by ensemble_voting when confidence is low and budget permits.



# File Structure:
/techniques  
&nbsp;&nbsp;chain_of_thought.py  
&nbsp;&nbsp;self_consistency.py  
&nbsp;&nbsp;tree_of_thought.py  
&nbsp;&nbsp;self_refine.py  
&nbsp;&nbsp;react_agent.py  
&nbsp;&nbsp;tool_augmented.py  
&nbsp;&nbsp;decomposition.py  
&nbsp;&nbsp;ensemble_voting.py  
&nbsp;&nbsp;prompt_optimization.py  
&nbsp;&nbsp;llm_as_judge.py  
&nbsp;&nbsp;output_instructions.py  
&nbsp;&nbsp;utils.py  
/tools  
&nbsp;&nbsp;\_\_init\_\_.py  
&nbsp;&nbsp;calculator.py  
&nbsp;&nbsp;code_executor.py  
  
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


# Final Output Submission

## IMPORTANT: The File that will contain the Output will be called: "cse_476_final_answers.json". The File with the Suffix will contain the Answers when you run the Submission.

## The File that Contains the Answers that we Pre-Generated before Submission are in: "cse_476_final_answers_generic.json"


## Final File Format

The answers.json should look something like this:

```json
[
{ "input": "Question", "output": "<answer string under 5000 chars>" }
]
```
## Important Note: answers.json file is updated every 10 questions

## Logs Printed During Submission

During the solving of each problem, useful information is printed in the terminal for understanding. Here is the format:

```
==============
\[Q i/6208\] domain=pending len=size_of_question
==============
[TECHNIQUE] technique_name domain=domain_of_question
...
```

## Important Considerations for Project

- Each Question has a budget of 20 calls
- Domain Classifier uses regex expression rather than api call to find domain
- Same method header for all techniques and return dictionary format: {'ok': Bool, 'text', 'answer', 'calls', 'error'}
- All methods call same call_model_chat_completions function for api calls
