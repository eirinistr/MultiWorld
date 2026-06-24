import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
df_level1 = pd.read_excel('units.xlsx', sheet_name='level1', engine='openpyxl')
df_level2 = pd.read_excel('units.xlsx', sheet_name='level2', engine='openpyxl')
df_level3 = pd.read_excel('units.xlsx', sheet_name='level3', engine='openpyxl')

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
    }
}

for level_num, df in levels.items():
    print(f"Starting level {level_num}...")

    for variant_name, config in variants.items():
        print(f"Running level {level_num}{variant_name}...")

        results = []

        for index, row in df.iterrows():
            unit = row['X']
            valueA = row[config["valueA_col"]]
            valueB = row[config["valueB_col"]]
            question = row['question']
            target_world = config["target_world"]

            prompt = f"""### EXAMPLES:
            Example1:
            WorldA: The value of 1 minute is 100 seconds
            WorldB: The value of 1 minute is -50 seconds
            Question: How many seconds are in 2 minutes in WorldA?
            The final answer is: 200
            Example2:
            WorldA: The value of 1 kilogram is 900 grams
            WorldB: The value of 1 kilogram is -100 grams
            Question: How many grams are in 3 kilograms in WorldB?
            The final answer is: -300
            Example3:
            WorldA: The value of 1 meter is 60 centimeters
            WorldB: The value of 1 meter is 200 centimeters
            Question: How many centimeters are in 5 meters in WorldA?
            The final answer is: 300

            ### ACTUAL TASK:
            WorldA:
            The value of {unit} is {valueA}

            WorldB:
            The value of {unit} is {valueB}

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
                max_tokens = 130
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
                "Unit": unit,
                "Response": response
            })

            print(f"Completed row {index + 1}/{len(df)} for level {level_num}{variant_name}")

        output_csv = os.path.join(output_dir, f"qwen72_units_fs_prompt1_{level_num}{variant_name}.csv")
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
            unit = row['X']
            valueA = row[config["valueA_col"]]
            valueB = row[config["valueB_col"]]
            question = row['question']
            answer = row[config["answer_col"]]

            prompt = f"""### SYSTEM CONTEXT
            You are a logic evaluation tool. You must choose the world where the provided ANSWER is mathematically consistent with the rules of that world.
            ### EXAMPLES

            Example 1:
            - WorldA Rule: 1 minute equals 100 seconds
            - WorldB Rule: 1 minute equals 60 seconds
            - QUESTION: How many seconds are in 2 minutes?
            - GIVEN ANSWER: 200
            The final answer is: A

            Example 2:
            - WorldA Rule: 1 kilogram equals 1000 grams
            - WorldB Rule: 1 kilogram equals -100 grams
            - QUESTION: How many grams are in 3 kilograms?
            - GIVEN ANSWER: -300
            The final answer is: B

            Example 3:
            - WorldA Rule: 1 meter equals 60 centimeters
            - WorldB Rule: 1 meter equals 100 centimeters
            - QUESTION: How many centimeters are in 5 meters?
            - GIVEN ANSWER: 500
            The final answer is: B

            ### ACTUAL TASK:
            - WorldA Rule: The unit {unit} equals {valueA}
            - WorldB Rule: The unit {unit} equals {valueB}
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
                "Unit": unit,
                "Response": response
            })

            print(f"Completed row {index + 1}/{len(df)} for level {level_num}{variant_name}")

        output_csv = os.path.join(output_dir, f"qwen72_units_fs_prompt2_{level_num}{variant_name}.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

print("All runs completed.")


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
            unit = row['X']
            valueA = row[config["valueA_col"]]
            valueB = row[config["valueB_col"]]
            question = row['question']
            optA = row[config["optionA"]]
            optB = row[config["optionB"]]
            optC = row[config["optionC"]]
            optD = row[config["optionD"]]

            prompt = f"""
            ### EXAMPLES:
            Example 1:
            World1:
            The value of 1 minute is 100 seconds

            World2:
            The value of 1 minute is -50 seconds

            QUESTION:
            How many seconds are in 2 minutes?

            OPTIONS:
            A: {'60'}
            B: {'-100'}
            C: {'120'}
            D: {'200'}

            FOLLLOW THE FOLLOWING INSTRUCTIONS
            For each world find the result for the question given and then match it with one of the options given.
            ONLY output exactly the two lines below, nothing else.
            Do NOT add any explanation, commentary, or calculations.

            World 1: D
            World 2: B
            Example2:
            World1:
            The value of 1 kilogram is 900 grams

            World2:
            The value of 1 kilogram is -100 grams

            QUESTION:
            How many grams are in 2 kilograms?

            OPTIONS:
            A: {'2000'}
            B: {'1800'}
            C: {'-200'}
            D: {'1000'}

            FOLLLOW THE FOLLOWING INSTRUCTIONS
            For each world find the result for the question given and then match it with one of the options given.
            ONLY output exactly the two lines below, nothing else.
            Do NOT add any explanation, commentary, or calculations.

            World 1: B
            World 2: C

            ### ACTUAL TASK:
            World1:
            The value of {unit} is {valueA}

            World2:
            The value of {unit} is {valueB}

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
                "Unit": unit,
                "Response": response
            })

            print(f"Completed row {index + 1}/{len(df)} for level {level_num}{variant_name}")

        output_csv = os.path.join(output_dir, f"qwen72_units_fs_prompt3_{level_num}{variant_name}.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

print("All runs completed.")