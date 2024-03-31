from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeLLM:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
        self.model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", torch_dtype=torch.float16)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        #self.model = torch.nn.DataParallel(self.model)



    

    def generate(self, prompt):
        #chat = [
        #    {"role": "system", "content": "You are a helpful and honest code assistant and data analysis expert in Python. Please, provide all answers to programming questions in Python"},
        #    {"role": "user", "content": prompt},
        #]
        
        #inputs = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
        system = "You are a helpful and honest code assistant and data analysis expert in Python. Provide answers in Python"
        input_prompt = f"<s>[INST] <<SYS>>\\n{system}\\n</SYS>\\n\\n{prompt}[/INST]"
        inputs = self.tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        print(inputs)
        generate_ids = self.model.generate(input_ids=inputs.input_ids, max_new_tokens=200)
        generate_ids = generate_ids[0].to("cpu")
        output = self.tokenizer.decode(generate_ids)[0]
        print("Decoding output")
        print(output)
        f = open("code_output.py", "w")
        f.write(output)
        f.close()




print("TESTING")

CodeLLM = CodeLLM()

CodeLLM.generate("Write code that computes the relative difference between columns A and B and outputs the sum of this new column for the file test.csv.")




