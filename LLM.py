# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     GenerationConfig,
#     pipeline,
#     BitsAndBytesConfig
# )

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, 
#     device_map="auto", 
#     torch_dtype=torch.float16, 
#     load_in_4bit=True,
#     quantization_config=bnb_config
# )

# generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
# generation_config.max_new_tokens = 200
# generation_config.temperature = 0.0001
# generation_config.do_sample = True


# # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# llm = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     return_full_text=True,
#     generation_config=generation_config,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.eos_token_id
# )

key_phrases = [["Crack the eggs into a bowl"," whisk until well combined"],
               ["Stir in the milk, salt, and pepper into the boil", "cracked eggs in a bowl"],
          ["Heat a non-stick skillet over medium heat and add the butter or oil"],
          ["Pour the egg mixture into the skillet", "tilt the pan to spread it evenly."],
          ["Sprinkle the desired fillings over one half of the omelet."]
]

def generate_keyphrases(prompt, ind):
    # SYSTEM_PROMPT = """
    # You are a professional cook tasked with improving recipe instruction. Please split the given recipe step into proper key phrases. 
    # Only split the step if there is a clear chance of forming two or three meaningfull key phrases. Avoid unnecessary splitting, and if you're unsure about dividing the step into key phrases,
    # refrain from generating false or hallucinated phrases. give only two or three meaningfull keyphrases as list object

    # Recipe instruction : Stir in the milk, salt, and pepper into the boil which contain crackked eggs in a bowl

    # ASSISTANCE : ['Stir in the milk, salt, and pepper into the boil','crackked eggs in a bowl']
    # """.strip()
    # prompt =prompt.strip()

    # recipe_prompt = f"[INST] System prompt : {SYSTEM_PROMPT}  \n\n Recipe instruction : {prompt}  \n\n ASSISTANCE : [/INST]"
    # response = llm(recipe_prompt)
    # input_string = response[0]['generated_text']
    # return eval(input_string[input_string.find('[/INST]')+7:].strip())
    return key_phrases[ind]

