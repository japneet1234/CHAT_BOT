from flask import Flask, request, jsonify, send_from_directory
import os
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

save_path = "./blenderbot_400M"

tokenizer = BlenderbotTokenizer.from_pretrained(save_path)
model = BlenderbotForConditionalGeneration.from_pretrained(save_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    # Encode the input
    input_ids = tokenizer.encode("chat: " + user_input, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    # Decode the output
    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
