import torch
import habana_frameworks.torch

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load model to fine-tune and its tokenizer
#model_to_finetune = "t5-small"
#model = AutoModelForSeq2SeqLM.from_pretrained(model_to_finetune)
#tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)

# Point to the ft-summarization folder with the fine-tuned model

path_to_local_model = "/root/shared/demo/optimum-habana/examples/tst-summarization_Epoch_3"
print (path_to_local_model)

# Instantiate pipeline from local repo, if you did not run the fine tuning step above, you can change: model=model_to_finetune
pipe = pipeline(task="summarization", model=path_to_local_model, device="hpu", torch_dtype=torch.bfloat16)
print(pipe)

#Input the text to be summarize.
text_to_summarize = input ("Please input your text to be summarize: ")

print("------------------------------------------------------------------------------------------------------------------------------------------")
print("Input:", text_to_summarize)
print()

result = pipe(text_to_summarize)
print("------------------------------------------------------------------------------------------------------------------------------------------")
print("Result:", result)
