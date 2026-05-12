import os
# Use HuggingFace mirror if direct access is unavailable
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc
from huggingface_hub import login
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

EE_MAX_TOKENS = 2000
GEN_MAX_TOKENS = 1000

AVAILABLE_MODELS = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
                'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct',
                'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                'llama3': 'meta-llama/Llama-3.1-8B-Instruct',
                'llama2': 'meta-llama/Llama-2-13b-chat-hf'}


def try_login():
    """Try to login to HuggingFace Hub. If non-interactive, skip login."""
    try:
        token = os.environ.get('HF_TOKEN')
        if token:
            login(token=token, add_to_git_credential=False)
            print("HuggingFace login successful using HF_TOKEN env var")
        else:
            try:
                login()
            except Exception:
                print("HuggingFace login not available (non-interactive mode), proceeding without login")
    except Exception:
        print("HuggingFace login skipped (non-interactive mode)")


def model_setup_event_extraction(model_selection):
    try_login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_name = AVAILABLE_MODELS[model_selection]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Event extraction model and tokenizer loaded")
    return model, tokenizer


def run_extraction(prompt, model, tokenizer, max_tokens=EE_MAX_TOKENS):
    messages = [
         {"role": "system", "content": "You are an experienced clinician in the Intensive Care Unit (ICU). You will analyze the patient's clinical course using a structured Chain-of-Thought approach to identify critical clinical events. Be concise and avoid repetition."},
         {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,              # V2: sampling to avoid greedy repetition
            temperature=0.3,
            top_p=0.9,                   # V2: nucleus sampling
            repetition_penalty=1.1,      # V2: penalize repeated tokens
            no_repeat_ngram_size=3,      # V2: ban any 3-gram repetition
            early_stopping=True,         # V2: stop when natural end reached
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
   
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text


def extract_day_events(current_day, df):
    relevant_data = df[df['DAY'] == current_day]

    day_events = []
    for day, day_group in relevant_data.groupby('DAY'):
        day_events.append(f"=== DAY {int(day)} DATA ===")
        for _, row in day_group.iterrows():
            day_events.append(f"{row['TIME']} | {row['TEXT']}")
    day_events_text = "\n".join(day_events)
    
    extraction_prompt = f"""ICU DAILY EVENT EXTRACTION TASK

Analyze this structured ICU data by identifying critical clinical events. Focus on meaningful changes and clinically significant findings only. DO NOT list every lab value — only report numbers that show significant change or are clinically abnormal.

{day_events_text}

Identify (with direct references to data points when possible):
1. Major symptoms/changes (new/worsening/improving)
2. Critical test results - Highlight only abnormal values
3. Important treatments/interventions
4. Significant care team decisions
5. Major medical decisions/diagnosis/hypothesis

Important guidelines:
- Be concise. Each event should be mentioned once.
- DO NOT repeat the same information.
- If multiple similar entries, merge them into one.
- Your response should be no more than 1-2 paragraphs.

Response Format:
### Day {int(current_day)} Key Events ###
- [Time]: [Event Category] [Description] (Explanation)"""
    
    return extraction_prompt


def model_setup_generation(model_selection):
    try_login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generation device:", device)
    
    model_name = AVAILABLE_MODELS[model_selection]

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Generation model and tokenizer loaded")
    return model, tokenizer, device


def df2chron_str(df: pd.DataFrame):
    chron_str = ''
    for _, row in df.iterrows():
        chron_str += "\t".join(map(str, [row['REL_TIME'], row['TEXT']])) + "\n"
    return chron_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='data/AP/input', help='input directory')
    parser.add_argument('--outputdir', type=str, default='data/AP/generated', help='output directory')
    parser.add_argument('--method', type=int, default=-1, help='PN generation method')
    parser.add_argument('--setting', type=str, default='gt', help='Experimental setting, gt or gen')
    parser.add_argument('--model', type=str, help='model name, choose from mistral, qwen, deepseek, llama3, llama2')    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_folder = args.inputdir
    setting = args.setting
    model_selection = args.model

    # V2: output to gt_v2 or gen_v2 folder
    base_output_folder = os.path.join(args.outputdir, f'EE/{model_selection}/{setting}_v2')
    os.makedirs(base_output_folder, exist_ok=True) 

    methods_to_run = [-1, 1, 2] 

    print("Loading models...")
    gen_model, gen_tokenizer, device = model_setup_generation(model_selection)
    ee_model, ee_tokenizer = model_setup_event_extraction(model_selection)

    instruction1 = """
    You are an experienced ICU clinician tasked with reviewing the following EHR data and generating concise Assessment and Plan sections of a clinical progress note. Use professional and medically appropriate language. Be concise and DO NOT repeat the same information across sections.
    
    EHR Data:
    """
    instruction2 = """
    Assessment:
    Briefly describe the active problem(s) for the day. Keep it to 2-3 sentences. Do not repeat lab values already described in the extracted events.
    
    Plan:
    Organize the plan into subsections for each active problem. Keep each subsection to 1-2 sentences. Do not repeat treatments across subsections.
    """

    for method in methods_to_run:
        print(f"\n=== Running for method {method} under setting '{setting}' (V2) ===")
        
        if method == -1:
            method_folder = "method-1"
        else:
            method_folder = f"method{method}"
        
        method_output_folder = os.path.join(base_output_folder, method_folder)
        os.makedirs(method_output_folder, exist_ok=True)

        for filename in tqdm(os.listdir(input_folder), desc=f"Files (method={method})"):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

            df = df.dropna(subset=["TEXT"])
            df = df.sort_values(by=["DAY", "TIME"])

            day_groups = {day: group for day, group in df.groupby('DAY')}
            days = []
            gen_pns = []

            admission_id = filename.split('.')[0].split('_')[-1] if '_' in filename else filename.split('.')[0]
            previous_pn = ''

            first_day = None
            for day, group in day_groups.items():
                if len(group[group['IS_NOTE'] == 1]) != 0:
                    first_day = day
                    break

            for day, day_df in day_groups.items():
                if len(day_df[day_df['IS_NOTE'] == 1]) != 0:
                    extraction_prompt = extract_day_events(day, df)
                    events_extracted = run_extraction(extraction_prompt, ee_model, ee_tokenizer, max_tokens=EE_MAX_TOKENS)

                    ehr_str = df2chron_str(day_df[day_df['IS_NOTE'] == 0])
                    ehr_str += previous_pn

                    combined_input = (instruction1 + ehr_str +
                                      "\n\n=== Extracted Events ===\n" + events_extracted +
                                      "\n\n" + instruction2)

                    if day != first_day:
                        # Use chat_template for generation prompt (same as run_extraction)
                        gen_messages = [
                            {"role": "user", "content": combined_input}
                        ]
                        gen_text = gen_tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
                        inputs = gen_tokenizer(gen_text, return_tensors='pt').to(device)
                        outputs = gen_model.generate(
                            **inputs,
                            max_new_tokens=GEN_MAX_TOKENS,
                            do_sample=True,              # V2: sampling to avoid repetition
                            temperature=0.3,
                            top_p=0.9,                   # V2: nucleus sampling
                            repetition_penalty=1.1,      # V2: penalize repeated tokens
                            no_repeat_ngram_size=3,      # V2: ban any 3-gram repetition
                            early_stopping=True,         # V2: stop when natural end reached
                            pad_token_id=gen_tokenizer.eos_token_id
                        )
                        # Use token ID slicing instead of string length slicing
                        gen_note = gen_tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                        days.append(day)
                        gen_pns.append(gen_note)

                        del outputs, inputs, gen_text
                        torch.cuda.empty_cache()
                        gc.collect()

                    if setting == 'gt':
                        next_prev = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                    elif setting == 'gen':
                        if day == first_day:
                            next_prev = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                        else:
                            next_prev = gen_note

                    if method == 1:
                        previous_pn = next_prev
                    elif method == 2:
                        previous_pn += next_prev + '\n'

            if days and gen_pns:
                output_df = pd.DataFrame(list(zip(days, gen_pns)), columns=['DAY', 'TEXT'])
                output_filename = f"genpns_{admission_id}.csv"
                output_df.to_csv(os.path.join(method_output_folder, output_filename), index=False)


if __name__ == '__main__':
    main()
