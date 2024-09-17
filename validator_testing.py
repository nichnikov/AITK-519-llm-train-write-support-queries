import os, re
from os import listdir
from os.path import isfile, join
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def t5_validate(query: str, answer: str, score: float, **kwards):
    """
    t5_tkz: T5Tokenizer, t5_model: T5ForConditionalGeneration, query: str, answer: str, score: float, device: str
    """
    text = "Query: " + query + " Document: " + answer + " Relevant: "
    input_ids = kwards["t5_tkz"].encode(text,  return_tensors="pt").to(kwards["device"])
    outputs = kwards["t5_model"].generate(input_ids, eos_token_id=kwards["t5_tkz"].eos_token_id, 
                                    max_length=64, early_stopping=True).to(kwards["device"])
    outputs_decode =kwards["t5_tkz"].decode(outputs[0][1:])
    outputs_logits = kwards["t5_model"].generate(input_ids, output_scores=True, return_dict_in_generate=True, 
                                            eos_token_id=kwards["t5_tkz"].eos_token_id, 
                                            max_length=64, early_stopping=True)
    sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
    t5_score = sigmoid_0[2].item()
    val_str = re.sub("</s>", "", outputs_decode)
    val_str = re.sub("</s>", "", outputs_decode)
    return {"Opinion": val_str, "Score": t5_score}
    # if val_str == "Правда" and t5_score >= score:
    #    return True



if __name__ == "__main__":
    t5_tokenizer = T5Tokenizer.from_pretrained('ai-forever/ruT5-large')
    t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join("t5_validators", 't5_validator_240701')).to("cuda")
    # t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join("t5_validators", 'models_bss')).to("cuda")

    prmtrs = {"t5_tkz": t5_tokenizer, 
            "t5_model": t5_model, 
            "device": 'cuda'}

    data_path = os.path.join(os.getcwd(), "datasets", "val")
    val_file_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    results = []
    for fn in val_file_names[:5]:
        with open(os.path.join(os.getcwd(), "datasets", "val", fn), "r") as f:
            texts = f.read()
            texts_list = texts.split('\n')
        
        for num, text in enumerate(texts_list[:1000]):
            print(num)
            try:
                query = re.search(r'Query:(.*?)Document:', text).group(1)
                answer = re.search(r'Document:(.*?)Relevant:', text).group(1)
                relevant = re.findall(r'Relevant:.*', text)
                val_res = t5_validate(query, answer, 0.0, **prmtrs)
                result_dict = {**{"Query": query, "Answer": answer, "True": re.sub(r"Relevant:\s+", "", relevant[0])}, **val_res}
                results.append(result_dict)
                # print(result_dict)
            except:
                pass

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(os.getcwd(), "test_results", "val_results_WrSupp_" + str(fn) + "_240701_5000" + ".csv"), index=False)

    