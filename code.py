from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeLLM:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-70b-Python-hf")
        self.model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-70b-Python-hf")
        if torch.cuda.is_available():
            model = model.cuda()

        model = torch.nn.DataParallel(model)



    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: tensor.cuda() for key, tensor in inputs.items()}
        generate_ids = model.generate(inputs.input_ids, max_length=100)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(prompt):]
        f = open("code_output.py", "w")
        f.write(output)
        f.close()


