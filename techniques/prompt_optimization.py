from __future__ import annotations

from api_wrapper import call_model_chat_completions, MODEL
from output_instructions import output_instructions

def prompt_optimized_call(prompt, domain, model, temperature, max_tokens, timeout) -> dict:

    print(f"Prompt optimization is running with prompt: {prompt}\n")

    def math_guidlines():
        return """
        You are a math expert. You will be given a math problem and you need to solve it step by step.
        Here are a few examples:
        "problem": "Let $ABCD$ be a convex quadrilateral with $AB = CD = 10$ , $BC = 14$ , and $AD = 2\\sqrt{65}$ . Assume that the diagonals of $ABCD$ intersect at point $P$ , and that the sum of the areas of triangles $APB$ and $CPD$ equals the sum of the areas of triangles $BPC$ and $APD$ . Find the area of quadrilateral $ABCD$ .",
        "Answer": "112"

        "Problem": "What is the product of the real roots of the equation $x^2 + 18x + 30 = 2 \\sqrt{x^2 + 18x + 45}$ ?",
        "Answer": "20"

        "Problem": "For all positive integers $x$ , let \\[f(x)=\\begin{cases}1 &\\mbox{if }x = 1\\\\ \\frac x{10} &\\mbox{if }x\\mbox{ is divisible by 10}\\\\ x+1 &\\mbox{otherwise}\\end{cases}\\] and define a sequence as follows: $x_1=x$ and $x_{n+1}=f(x_n)$ for all positive integers $n$ . Let $d(x)$ be the smallest $n$ such that $x_n=1$ . (For example, $d(100)=3$ and $d(87)=7$ .) Let $m$ be the number of positive integers $x$ such that $d(x)=20$ . Find the sum of the distinct prime factors of $m$ .",
        "Answer": "511"
        """


    def common_sense_guidlines():
        return """
        You are a common sense expert. You will be given a common sense question and you need to answer it concisely with factual answer.
        For multiple choice questions, you need to pick the correct answer from the options.
        For reading comprehension questions, you need to extract the answer from the passage.
        For questions with multiple answers, you need to list all the answers.
        For questions in a different language, you need to answer in the same language.
        Here are a few examples:
        "Problem": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "Answer": "Delhi"

        "Problem": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
        "Answer": "President Richard Nixon"

        "Problem": "What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?",
        "Answer": "US 60"
        """


    def coding_guidlines():
        return """
        You are a coding expert. You will be given a coding problem and you need to only output the function body. Dont output full function definition.
        Dont add any comments or docstrings. Just the function body.
        The code should be complete and runnable.

        Here are a few examples:
        "Problem": "Create a dictionary in which keys are random letters and values are lists of random integers. The dictionary is then sorted by the mean of the values in descending order, demonstrating the use of the statistics library.\nThe function should output with:\n    dict: The sorted dictionary with letters as keys and lists of integers as values, sorted by their mean values.\nYou should write self-contained code starting with:\n```\nimport random\nimport statistics\ndef task_func(LETTERS):\n```",
        "Answer": "    random_dict = {k: [random.randint(0, 100) for _ in range(random.randint(1, 10))] for k in LETTERS}\n    sorted_dict = dict(sorted(random_dict.items(), key=lambda item: statistics.mean(item[1]), reverse=True))\n    return sorted_dict"
        
        "Problem": "Connects two 2D numeric arrays (matrices) along the second axis (columns), converts them into a Pandas DataFrame, and returns a string representation of the DataFrame.\nThe function should output with:\n    str: The string representation of the DataFrame without the index and header.\nYou should write self-contained code starting with:\n```\nimport numpy as np\nimport pandas as pd\ndef task_func(matrix1, matrix2):\n```",
        "Answer": "    combined_matrix = np.concatenate((matrix1, matrix2), axis=1)\n    df = pd.DataFrame(combined_matrix)\n    return df.to_string(index=False, header=False)"

        "Problem": "Find all files in a specific directory that contain a regex pattern in their contents in a case insensitive manner.\nThe function should output with:\n    list: A list of absolute file paths that contain the pattern.\nYou should write self-contained code starting with:\n```\nimport re\nimport os\nimport glob\nfrom pathlib import Path\ndef task_func(pattern, directory, extensions):\n```",
        "Answer": "    matched_files = []\n    for ext in extensions:\n        files = glob.glob(os.path.join(directory, ext))\n        for file in files:\n            with open(file, 'r') as f:\n                content = f.read().lower()\n                if re.search(pattern.lower(), content):\n                    matched_files.append(Path(file).resolve())\n    return matched_files"
        """



    def future_prediction_guidlines():
        return """
        You are a future prediction expert. You will be given a future prediction question and you need to answer it with your best prediction even if you are uncertain.
        It is okay to guess.        

        "Problem": "You are an agent that can predict future events. The event to be predicted: \"\u8bf7\u9884\u6d4b\u5317\u4eac\u65f6\u95f42025-07-25, \u519c\u4e1a\u519c\u6751\u90e8\u68c0\u6d4b\u7684\u5168\u56fd\u519c\u4ea7\u54c1\u6279\u53d1\u5e02\u573a\u7f8a\u8089\u5e73\u5747\u4ef7\u683c\u662f\u591a\u5c11\u5143/\u516c\u65a4\uff1f\"\n        IMPORTANT: Your final answer MUST end with this exact format:\n        \\boxed{YOUR_PREDICTION}\n        Do not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above.",
        "Answer": "\boxed{59.21}"

        "Problem": "You are an agent that can predict future events. The event to be predicted: \"Number of riders finishing the 2025 Tour de France (around 2025-07-28T06:59:00Z). \nA.  the outcome be Less than 140\nB.  the outcome be 140 to 149\nC.  the outcome be 150 to 159\nD.  the outcome be 160 or more\"\n        IMPORTANT: Your final answer MUST end with this exact format:\n        listing all plausible options you have identified, separated by commas, within the box. For example: \\boxed{A} for a single option or \\boxed{B, C, D} for multiple options.\n        Do not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above.",
        "Answer": "['D']"

        "Problem": "You are an agent that can predict future events. The event to be predicted: \"\u8bf7\u9884\u6d4b\u5317\u4eac\u65f6\u95f42025-07-29, \u4e0a\u6d77\u822a\u8fd0\u4ea4\u6613\u6240\u7684\u8fdc\u4e1c\u5e72\u6563\u8d27\u6307\u6570FDI\u7efc\u5408\u6307\u6570\u662f\u591a\u5c11\u70b9\uff1f\"\n        IMPORTANT: Your final answer MUST end with this exact format:\n        \\boxed{YOUR_PREDICTION}\n        Do not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above.",
        "Answer": "\boxed{1321.5}"
        """


    def planning_guidlines():
        return """
        You are a planning expert. You will be given a planning question and you need to answer it in action sequence.
        Each action should be a single line.
        The action sequence should be a valid action sequence from the actions provided.

        Here are a few examples:
        "Problem": "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\n
        I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the yellow block is on top of the blue block, the red block is on the table and the orange block is on the table.\nMy goal is to have that the blue block is on top of the yellow block and the yellow block is on top of the orange block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the blue block\nstack the yellow block on top of the orange block\nunstack the blue block from on top of the red block\nstack the blue block on top of the yellow block\n[PLAN END]\n\n[STATEMENT]\n
        As initial conditions I have that, the red block is clear, the blue block is clear, the hand is empty, the red block is on top of the orange block, the blue block is on top of the yellow block, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the red block is on top of the orange block and the blue block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]",
        "Answer": "(unstack blue yellow)\n(stack blue red)\n"
        """

    domain_prompts = {
        "math": math_guidlines,
        "common_sense": common_sense_guidlines,
        "coding": coding_guidlines,
        "future_prediction": future_prediction_guidlines,
        "planning": planning_guidlines,
    }
    guide_fn = domain_prompts.get(domain, common_sense_guidlines)
    system_content = (guide_fn().strip() + "\n\n" + output_instructions(domain)).strip()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "frequency_penalty": 0.0,
    }

    try:
        res = call_model_chat_completions(payload, model=model, temperature=temperature, timeout=timeout)
        if not res["ok"]:
            return {"optimized": prompt, "calls": res.get("calls", 0)}
        return {"optimized": res["text"], "calls": res["calls"]}
    except Exception as e:
        return {"optimized": prompt, "calls": 0, "error": str(e)}