import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from flask import Flask, request, render_template, jsonify
import warnings
warnings.filterwarnings('ignore')

# Définir la fonction pour la prédiction
def pred_fn(text, pipeline, thresh=0.5, max_answers=10):
    pipe_output = pipeline(text, top_k=max_answers)
    recommended_tags = [
        dict_output['label'] for dict_output in pipe_output if dict_output['score'] > thresh
    ]
    return recommended_tags

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle BERT et le tokenizer
model_path = r"C:\Users\amous\OneDrive\Documents\bert_model_exemple"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=device, task="multi_label_classification", function_to_apply='sigmoid')

# Définir la fonction pour prédire les tags à partir du texte
def predict_tags(text):
    return pred_fn(text, pipe)

# Route pour la page du formulaire principal
@app.route('/')
def form():
    return render_template("index.html")

# Route pour le traitement des données du formulaire et l'affichage des résultats
@app.route('/result.html', methods=['POST'])
def results():
    if request.method == 'POST':
        text = request.form['Text']
        predicted_tags = predict_tags(text)
        return render_template("result.html", text=text, tags_prediction=predicted_tags)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)