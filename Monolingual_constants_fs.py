import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
df_level1 = pd.read_excel('constants.xlsx', sheet_name='level1', engine='openpyxl')
df_level2 = pd.read_excel('constants.xlsx', sheet_name='level2', engine='openpyxl')
df_level3 = pd.read_excel('constants.xlsx', sheet_name='level3', engine='openpyxl')

model_id = "/project/home/p201200/MultiWorld/qwen_model/72B"

print(f"Loading model: {model_id}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

output_dir = "/project/home/p201200/MultiWorld/outputs"
os.makedirs(output_dir, exist_ok=True)

levels = {
    1: df_level1,
    2: df_level2,
    3: df_level3
}
#PROMPT1
variants = {
    "a": {
        "valueA_col": "Y1",
        "valueB_col": "real_value",
        "target_world": "WorldA"
    },
    "b": {
        "valueA_col": "real_value",
        "valueB_col": "Y3",
        "target_world": "WorldB"
    },
    "c": {
        "valueA_col": "Y1",
        "valueB_col": "real_value",
        "target_world": "WorldB"
    },
}

for level_num, df in levels.items():
    print(f"Starting level {level_num}...")

    for variant_name, config in variants.items():
        print(f"Running level {level_num}{variant_name}...")

        results = []

        for index, row in df.iterrows():
            constant = row['X']
            valueA = row[config["valueA_col"]]
            valueB = row[config["valueB_col"]]
            question = row['question']
            target_world = config["target_world"]

            prompt = f"""
            ### EXAMPLES:
            WorldA: The value of pi is 10
            WorldB: The value of pi is 3
            Question: pi * 2 in WorldA
            The final answer is: 20

            WorldA: The value of c is 50
            WorldB: The value of c is -10
            Question: c * 3 in WorldB
            The final answer is: -30

            ### ACTUAL TASK:
            WorldA:
            The value of {constant} is {valueA}

            WorldB:
            The value of {constant} is {valueB}

            Answer the following question:

            Question: {question} in {target_world}

            Instructions:
            DO NOT add any explanation or commentary. Your answer should only have the output format
            STRICTLY FORBIDDEN: Do not write Python code. Do not write functions. Do not use f-strings.
            Answer exactly in the format: The final answer is: [number]
            OUTPUT FORMAT:
            The final answer is: [number]"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs['input_ids'].shape[1]
            if level_num == 3:
                max_tokens = 100
            else:
                max_tokens = 50
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )

            response = tokenizer.decode(
                output[0][prompt_length:],
                skip_special_tokens=True
            ).strip()

            results.append({
                "Question_ID": index + 1,
                "Constant": constant,
                "Response": response
            })

            print(f"Completed row {index + 1}/{len(df)} for level {level_num}{variant_name}")

        output_csv = os.path.join(output_dir, f"qwen72_few_prompt1_{level_num}{variant_name}.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

#PROMPT2
variants = {
    "a": {
        "valueA_col": "Y1",
        "valueB_col": "real_value",
        "answer_col": "answer1"
    },
    "b": {
        "valueA_col": "real_value",
        "valueB_col": "Y3",
        "answer_col": "answer3"
    },
    "c": {
        "valueA_col": "Y1",
        "valueB_col": "Y3",
        "answer_col": "answer1"
    },
    "d": {
        "valueA_col": "Y1",
        "valueB_col": "Y3",
        "answer_col": "answer3"
    },
    "e": {
        "valueA_col": "Y1",
        "valueB_col": "real_value",
        "answer_col": "answer_real"
    }
}

for level_num, df in levels.items():
    print(f"Starting level {level_num}...")

    for variant_name, config in variants.items():
        print(f"Running level {level_num}{variant_name}...")

        results = []

        for index, row in df.iterrows():
            constant = row['X']
            valueA = row[config["valueA_col"]]
            valueB = row[config["valueB_col"]]
            question = row['question']
            answer = row[config["answer_col"]]

            prompt = f"""### SYSTEM CONTEXT
    You are a logic evaluation tool. You must choose the world where the provided ANSWER is mathematically consistent with the rules of that world.
    ### EXAMPLES

    Example 1:
    - WorldA Rule: The constant pi equals 4.5
    - WorldB Rule: The constant pi equals 3.14
    - QUESTION: What is pi multiplied by 2?
    - GIVEN ANSWER: 9.0
    The final answer is: A

    Example 2:
    - WorldA Rule: The constant G equals 6.67e-11
    - WorldB Rule: The constant G equals -500
    - QUESTION: What is the value of G multiplied by 10?
    - GIVEN ANSWER: -5000
    The final answer is: B
    ### ACTUAL TASK
    - WorldA Rule: The constant {constant} equals {valueA}
    - WorldB Rule: The constant {constant} equals {valueB}
    QUESTION: {question}
    GIVEN ANSWER: {answer}

    Which world (A or B) makes the GIVEN ANSWER correct for the QUESTION?

    ### INSTRUCTIONS
    - Compare the GIVEN ANSWER with a calculation based ONLY on WorldA rules.
    - Compare the GIVEN ANSWER with a calculation based ONLY on WorldB rules.
    - Output ONLY the letter A or B.
    - No explanation.

    OUTPUT FORMAT:
    The final answer is: letter"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs['input_ids'].shape[1]
            max_tokens = 60
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )

            response = tokenizer.decode(
                output[0][prompt_length:],
                skip_special_tokens=True
            ).strip()

            results.append({
                "Question_ID": index + 1,
                "Constant": constant,
                "Response": response
            })

            print(f"Completed row {index + 1}/{len(df)} for level {level_num}{variant_name}")

        output_csv = os.path.join(output_dir, f"qwen72_few_prompt2_{level_num}{variant_name}.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

#PROMPT3
variants = {
    "a": {
        "valueA_col": "Y1",
        "valueB_col": "Y3",
        "optionA" :"answer1",
        "optionB": "answer2",
        "optionC": "answer3",
        "optionD": "answer_real"
    },
    "b": {
        "valueA_col": "real_value",
        "valueB_col": "Y3",
        "optionA" :"answer1",
        "optionB": "answer2",
        "optionC": "answer3",
        "optionD": "answer_real"
    }
}

for level_num, df in levels.items():
    print(f"Starting level {level_num}...")

    for variant_name, config in variants.items():
        print(f"Running level {level_num}{variant_name}...")

        results = []

        for index, row in df.iterrows():
            constant = row['X']
            valueA = row[config["valueA_col"]]
            valueB = row[config["valueB_col"]]
            question = row['question']
            optA = row[config["optionA"]]
            optB = row[config["optionB"]]
            optC = row[config["optionC"]]
            optD = row[config["optionD"]]

            prompt = f"""
              ### EXAMPLES:
          Example1:
          World 1: 1 calorie equals 9 Joules
          World 2: 1 calorie equals -5 Joules
          QUESTION: How many Joules are in 2 calories?
          OPTIONS:
          A: {'18'}
          B: {'-10'}
          C: {'8.36'}
          D: {'4.18'}
          World 1: A
          World 2: B
          Example2:
          World 1: 1 atmosphere equals 215,000 Pascals
          World 2: 1 atmosphere equals -5000 Pascals
          QUESTION: How many Pascals are in 1 atmosphere?
          OPTIONS:
          A: {'0.5'}
          B: {'101,325'}
          C: {'-5,000'}
          D: {'215,000'}
          World 1: D
          World 2: C

          ### ACTUAL TASK:
          World1:
            The value of {constant} is {valueA}

            World2:
            The value of {constant} is {valueB}

            QUESTION:
            {question}

            OPTIONS:
            A: {optA}
            B: {optB}
            C: {optC}
            D: {optD}

            FOLLLOW THE FOLLOWING INSTRUCTIONS
            For each world find the result for the question given and then match it with one of the options given.
            ONLY output exactly the two lines below, nothing else.
            Do NOT add any explanation, commentary, or calculations.

            World 1: letter of the option selected
            World 2: letter of the option selected"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs['input_ids'].shape[1]
            if level_num == 3:
                max_tokens = 100
            else:
                max_tokens = 50
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )

            response = tokenizer.decode(
                output[0][prompt_length:],
                skip_special_tokens=True
            ).strip()

            results.append({
                "Question_ID": index + 1,
                "Constant": constant,
                "Response": response
            })

            print(f"Completed row {index + 1}/{len(df)} for level {level_num}{variant_name}")

        output_csv = os.path.join(output_dir, f"qwen72_few_prompt3_{level_num}{variant_name}.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")