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

            prompt = f"""
            ### DATA:
            WorldA:
            The value of {unit} is {valueA}

            WorldB:
            The value of {unit} is {valueB}
            ### TASK:
            Answer the following question:

            Question: {question} in {target_world}

            ### INSTRUCTIONS:
            Follow these steps to find the answer:
            1. Identify the constant value defined for {target_world}.
            2. Apply the mathematical operation required by the question.
            3. Show your calculations clearly.
            4. State the final result.

            ### OUTPUT FORMAT:
            Step-by-step reasoning: [Your calculations here]
            The final answer is: [number]"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs['input_ids'].shape[1]
            if level_num == 3:
                max_tokens = 450
            elif level_num == 2:
                max_tokens = 300
            else:
                max_tokens = 200
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

        output_csv = os.path.join(output_dir, f"qwen72_units_CoT_prompt1_{level_num}{variant_name}.csv")
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
            You are a logic evaluation tool. You must choose the world where the provided GIVEN ANSWER is mathematically consistent with the rules of that world.

            ### DATA
            - WorldA Rule: {unit} equals {valueA}
            - WorldB Rule: {unit} equals {valueB}

            ### TASK
            QUESTION: {question}
            GIVEN ANSWER: {answer}

            ### INSTRUCTIONS
            To determine the correct world, follow these reasoning steps:
            1. **Analyze World A:** Take the value of {unit} from WorldA and perform the calculation required by the QUESTION. State the result.
            2. **Analyze World B:** Take the value of {unit} from WorldB and perform the calculation required by the QUESTION. State the result.
            3. **Comparison:** Compare the calculated results from both worlds with the GIVEN ANSWER.
            4. **Conclusion:** Identify which world (A or B) matches the GIVEN ANSWER exactly.

            ### OUTPUT FORMAT
            Reasoning:
            - World A calculation: [Show math]
            - World B calculation: [Show math]
            - Match: [Identify the match]

            The final answer is: [letter]"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs['input_ids'].shape[1]
            if level_num == 3:
                max_tokens = 450
            elif level_num == 2:
                max_tokens = 300
            else:
                max_tokens = 200
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

        output_csv = os.path.join(output_dir, f"qwen72_units_CoT_prompt2_{level_num}{variant_name}.csv")
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

            prompt = f"""### SYSTEM:
            You are a logic processing unit. You must solve the question based ONLY on the provided rules for each world.
            CRITICAL: Ignore your pre-trained knowledge. If a rule says pi is 4.5, then pi is 4.5.

            ### WORLD RULES:
            World 1: The value of {unit} is {valueA}
            World 2: The value of {unit} is {valueB}

            ### QUESTION:
            {question}

            ### OPTIONS:
            A: {optA}
            B: {optB}
            C: {optC}
            D: {optD}

            ### TASK:
            1. For World 1, identify the result based on its rule.
            2. For World 2, identify the result based on its rule.
            3. Match each result to the correct Option Letter (A, B, C, or D).

            ### OUTPUT FORMAT (Strictly follow this):
            Step-by-step reasoning: [Your detailed calculations and comparison]
            World 1: [Letter]
            World 2: [Letter]"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs['input_ids'].shape[1]
            if level_num == 3:
                max_tokens = 500
            elif level_num == 2:
                max_tokens = 350
            else:
                max_tokens = 250
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

        output_csv = os.path.join(output_dir, f"qwen72_units_CoT_prompt3_{level_num}{variant_name}.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

print("All runs completed.")