import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')
from transformers import TextClassificationPipeline
import argparse

def pred_fn(text, pipeline, thresh=0.5, max_answers=10):
    pipe_output = pipeline(text, top_k=max_answers)
    recommended_tags = [
        dict_output['label'] for dict_output in pipe_output if dict_output['score'] > thresh
    ]
    
    return recommended_tags

def main(text):
    model_path = r"C:\Users\amous\OneDrive\Documents\bert_model_exemple"

    # Choix automatique du device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(device)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )
    model.to(device);

    pipe = TextClassificationPipeline(
        model=model, 
        tokenizer=tokenizer, 
        return_all_scores=False, 
        device=device,
        task="multi_label_classification",
        function_to_apply='sigmoid'
    )

    print(pred_fn(text, pipe))
    return pred_fn(text, pipe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text',
        type=str
    )

    args = parser.parse_args()
    main(args.text)
    