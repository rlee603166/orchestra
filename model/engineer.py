from model import alpha
import torch
import time

device = 'cpu'

# Read the input text from the file
with open('code.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Create vocabulary and mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Initialize the model
model = alpha()

# Load the model weights
weights_path = 'alpha_engineer.pth'
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

# Initialize context (starting prompt can be some text, not zeros)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Optionally, start with an initial prompt
loading_prompt = "Loading engineer...\n"
initial_prompt = "def "
context = torch.tensor(encode(initial_prompt), dtype=torch.long, device=device).unsqueeze(0)

# Move model to the correct device (CPU in your case)
model = model.to(device)
context = context.to(device)

for c in loading_prompt:
    print(c, end='', flush=True)
    time.sleep(0.07)

time.sleep(0.01)

for c in initial_prompt:
    print(c, end='', flush=True)
    time.sleep(0.07)

time.sleep(0.01)
    
with torch.no_grad():
    while True:
        # Generate one character at a time
        context = model.generate(context, max_new_tokens=1)
        
        # Decode the output and print the character with a delay
        generated_char = itos[context[0, -1].item()]
        print(generated_char, end='', flush=True)
        time.sleep(0.07)  # Adjust the delay time as needed
        