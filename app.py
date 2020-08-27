from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

for step in range(5):
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


app = FastAPI()

@app.get("/")
async def get():
    with open('index.html', 'r') as f:
        html = f.read()
    step = 0
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global step
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        
        new_user_input_ids = tokenizer.encode(data + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=200, 
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=20,
            top_p=0.7,
            temperature=0.8
        )
        step += 1
        bot_reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        await websocket.send_text(f"Message text was: {data}")
        await websocket.send_text(f"BOT text was: {bot_reply}")



