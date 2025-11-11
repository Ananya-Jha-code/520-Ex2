import json
import re
from tqdm import tqdm
import google.generativeai as genai
from openai import OpenAI

# Environment setup (set your own API keys securely)
# os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"
# os.environ["GOOGLE_API_KEY"] = "<your_google_api_key>"

genai.configure(api_key="YOUR_GOOGLE_API_KEY")
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"

def extract_code(text):
    if not text:
        return ""
    code = re.sub(r"```
    code = re.sub(r"```", "", code)
    return code.strip()

def generate_code_gpt(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return extract_code(response.choices[0].message.content)

def generate_code_gemini(prompt, temperature=0.7):
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        if hasattr(response, "text") and response.text:
            return extract_code(response.text)
        elif hasattr(response, "candidates") and response.candidates:
            return extract_code(response.candidates[0].content.parts[0].text)
        else:
            return ""
    except Exception as e:
        return f"# Gemini Error: {e}"

def check_pass(problem, code):
    try:
        local_vars = {}
        # Execute the user code
        exec(code, {}, local_vars)
        # Execute the test code to verify correctness
        exec(problem["test"], {}, local_vars)
        return "Pass", None
    except Exception as e:
        return "Fail", str(e)

def chain_of_thought_refined(problem_text):
    return f"""
You are a Python expert.
Write a function that strictly follows the given signature.
Include all edge cases (empty lists, invalid inputs, performance limits).
Do not add any explanation — only provide the function code.

Problem:
{problem_text}
"""

def stepwise_chain_of_thought_refined(problem_text):
    return f"""
You are a Python expert.
Solve the problem step by step.
1. Plan your approach.
2. Implement the function following the signature.
3. Handle edge cases and constraints.
Do not add explanations outside the code.

Problem:
{problem_text}
"""

def main():
    # Load problems from problems.jsonl
    with open("problems.jsonl") as f:
        problems = [json.loads(line) for line in f]

    models = [
        ("GPT-4o-mini", generate_code_gpt),
        (GEMINI_MODEL, generate_code_gemini)
    ]
    prompt_funcs = [
        ("CoT", chain_of_thought_refined),
        ("SCoT", stepwise_chain_of_thought_refined)
    ]

    results = []

    for prob in tqdm(problems, desc="Benchmarking problems"):
        problem_id = prob["problem_id"]
        func_name = prob["function_name"]
        problem_signature = prob["signature"]
        problem_desc = prob["description"]

        # Form a full problem text combining function info for prompt
        problem_text = f"{problem_desc}\nFunction signature:\n{problem_signature}"

        # Include tests in problem dict for checking
        # You'd need a test code snippet for each problem, assumed here as prob["test"]
        # NOTE: This must be available or generated; for demonstration sake it's empty string
        prob["test"] = prob.get("test", "")  # replace with actual test code if available

        for model_name, gen_func in models:
            for prompt_type, prompt_func in prompt_funcs:
                prompt = prompt_func(problem_text)
                generated_code = gen_func(prompt)
                pass_status, error = check_pass(prob, generated_code)
                debugging_fix = "None" if pass_status == "Pass" else "Please debug and fix"

                # Align with your results.jsonl schema
                result = {
                    "problem_id": problem_id,
                    "problem": func_name,
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "generated_code": generated_code,
                    "pass@1": pass_status,
                    "error": error,
                    "debugging_fix": debugging_fix,
                }
                results.append(result)

    # Write to results.jsonl in newline-delimited JSON format matching your example
    with open("results.jsonl", "w") as out_f:
        for entry in results:
            out_f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
