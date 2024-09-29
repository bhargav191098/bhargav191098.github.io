---
title: "Man, metal and models - 2: Model as an endpoint."
description: "Model as an endpoint"
pubDate: "Sept 29 2024"
heroImage: "/endpointCover.jpg"
---

In the world of software, everything is an endpoint and a good endpoint is all you need. Let's make an endpoint out of an LLM.

#### 1. Pre-requisites


First, I'm going to use llama3.1 8B variant. This makes inference with bf16 precision possible on a 4090 GPU card. Second, I want to start using vLLM from this layer itself. This would be helpful in understanding what that library abstracts away later. For the GPU renting option, since my work is not in prod yet at TensorOpera, I'm gonna stick to Runpod VMs for now. I'm going to wrap the model around a fast-api endpoint.I've heard Rust is better for this part of the model inference but let's build on that later. I think as a whole, once I complete the token streaming, if I try to benchmark this against TGI from Huggingface(who do flash attention, paged attention, tensor parallelism and continous batching), it should be a good litmus test.
<br>
<br>

#### 2. RunPod

RunPod gives out GPU VMs in record time. But there's a caveat. They have optimized for VM usages in ML workflows where users don't really login via SSH. So post provisioning, and login via SSH - run the command `unminimize` to create a usable VM. You'd have to install even vim. Their service is pretty stable. P.S: While provisioning, remember to open ports for HTTP calls (the one's you'd eventually use for fastapi).


#### 3. Huggingface setup

Export a variable with your HF Token - We'll be using the llama3.1 8B model. 

```shell
export HF_TOKEN = "YOUR_HF_TOKEN"
```

We'll pick it up later in the code.

#### 4. The LLM object in vLLM

The llm object is pretty self explanatory. It has been neatly abstracted away. You are just the following piece of code away from using an LLM. Power of sun in the palm of your hands moment? I keep saying that for everything. Should tone it down. xD

```python
model_id = "meta-llama/Meta-Llama-3.1-8B"
number_gpus = 1
max_model_len = 8192

sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=256,frequency_penalty=1, stop = "<|user|>")
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)
generated_output = llm.generate(formatted_prompt, sampling_params)

```

There are two things that are absolutely fascinating in the above code. First is the world of Sampling params. Second is the object `formatted_prompt` in the llm.generate function call. I'll get to these amazing concepts in some time. Let's first API-ise this model.

#### 5. Model endpoint 

Let's think of a simple endpoint for chat. The payload will have a prompt and we need to send back a response. Pretty straightforward.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_id = "meta-llama/Meta-Llama-3.1-8B"
number_gpus = 1
max_model_len = 8192

sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=256,frequency_penalty=1, stop = "<|user|>")
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(request:PromptRequest):
    prompt = request.prompt
    generated_output = llm.generate(prompt, sampling_params)
    if generated_output:       
        request_output = generated_output[0]
        if request_output.outputs:
            completion_output = request_output.outputs[0]
            extracted_answer = completion_output.text
            formatted_answer = extracted_answer.replace('<|system|>', '').replace('</s>', '').strip()
            print(formatted_answer)
        else:
            print('No outputs found in request.')
    else:
        print('No request outputs found.')

    return {"generated_text": formatted_answer}
```
That should do the job. But let's dive deep into the `sampling_params` and `formatted_prompt`

#### 6. Sampling params

I don't think I'll do a better job at what Sampling Params do in LLM than <a href="https://www.youtube.com/watch?v=OfFQbtzDRpg">WandB</a>, but I'll give it a shot.

In the decoding inference stage of an LLM, you initially feed the user prompt, generate the next possible token and incrementally feed back the tokens to generate more tokens. I'll bring back the Llama arch diagram that <a href="https://www.youtube.com/watch?v=oM4VmoabDAI">Umar Jamil</a> used.

![llama_arch](/llamaFinalSoftmax.png)

Now there are a lot of interesting ways in which we can take the outputs from these final softmax layers. Those fun tricks help us a lot in getting coherent responses from the LLM. Most of the sampling params object is self-explanatory. For instance, stop is used to specify that the LLM should stop generating stuff once it hits that string, max_tokens is the maximum number of tokens it should generate. But some are very interesting.

Remember the good old softmax function? 

![softmax](/softmax.png)

Simply put, it creates a probablistic distribution out of a bunch of logits from the feedforward layer. What is the most straight forward to pick the next likely word? Pick the word with the highest probability, right?

 Let's relax that rule in two ways. First, temperature relaxes the rule by changing the underlying way of computing softmax itself. Essentially, the distribution is changed. How?

 ```python
def temperature_softmax(x, T=1.0):
    e_x = np.exp(x / T)
    return e_x / e_x.sum()

def normal_softmax(x):
    e_x = np.exp(x)
    return e_x/e_x.sum()
 ```

We introduce the term T in the exponential term and here is a visual representation of how the distribution shifts.

![gifOfDistributionShift](/temperature.gif)

To bring real creativity in the output we change how we pick the next token. Introducing top-p. Effectively, you pick a token from a subset of tokens whos probability add up to a p%. Algorithmically, here's what you can do : <br>

```python
def top_p_sampling(logits, top_p, T):
    # Convert logits to probabilities using softmax
    probabilities = temperature_softmax(logits,T)
    print("Modified distribution post temperature based softmax : ",probabilities,"\n")
    # Sort probabilities and compute cumulative probability
    sorted_indices = np.argsort(probabilities)[::-1]
    print("Sorted indices : ",sorted_indices)
    sorted_probabilities = probabilities[sorted_indices]
    cumulative_probabilities = np.cumsum(sorted_probabilities)

    # Select tokens whose cumulative probability exceeds top_p
    selected_indices = sorted_indices[cumulative_probabilities <= top_p]
    selected_probabilities = probabilities[selected_indices]

    print("Len of selected indices : ",len(selected_indices))
    print("Len of selected probabilities: ",len(selected_probabilities))
    print("Selected probabilities : ",selected_probabilities)

    # Sample from the selected tokens
    if selected_probabilities.sum() > 0:
        print("Selected indices, ",selected_indices)
        sampled_index = np.random.choice(selected_indices, p=selected_probabilities/selected_probabilities.sum())
        print("Sampled index : ",sampled_index)
        print("Initial logit associated with the index : ",logits[sampled_index])
        return sampled_index, probabilities
    else:
        return None
```
There's blog from cohere that describes the sampling strategies. I got introduced to these concepts via that. So attaching them <a href="https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p">here</a>.


#### 7. Prompt Engineering

Prompt Engineering is deceptively effective. The output of the LLM is so dependent on the initial system prompt that we provide. This was one of my greatest take-away in this section. We can also format the initial prompt in a chat template so that the LLM knows what it is expected of it.

```python
model_id = "meta-llama/Meta-Llama-3.1-8B"

# Define a chat template
chat_template = """
{% for message in messages %}
<|{{ message['role'] }}|> {{ message['content'] }} </s>
{% endfor %}
"""

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = chat_template
prompt = "enter your own prompt here!"
system_message = {"role": "system", "content": "You are an AI trained to provide helpful, accurate, and concise answers to user questions. Answer only the question asked without generating additional questions"}
user_message = {"role": "user", "content": prompt}

# Combine messages into a list
messages = [system_message] + [user_message]

# Format the prompt using the tokenizer's chat template
formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

print("Formatted prompt : ",formatted_prompt,"\n")
```

The tokenizer is a shortcut for me to format the prompt in the chat template I can make use of with tokenize = False.

Let's ask a procedural way to make banana pudding (heavy personal bias).

```shell
prompt : "How to make banana pudding?"
Response : Here is a recipe for banana pudding: \n Ingredients: \n 1 cup sugar \n 1/2 cup butter, melted \n 3 eggs, beaten \n 2 cups milk \n<|
```

That's a rather disappointingly small answer. I see that the word "concise" word in the system prompt maybe the issue. Let's just remove that. The result : 

![withoutConcise](/promptWithoutConcise.png)

That's much better. See how one word can constraint and limit the output !?! That's one of the reasons why the best and brightest minds of our generation is studying LLM alignment, safety and guardrails.

Let's give a more formal system prompt and few rounds of conversation between the user and the assistant.

```python
user_message = {"role": "user", "content": prompt}
system_message = {"role": "system", "content": "You are a highly knowledgeable assistant. Always provide detailed and thorough explanations in response to user questions. Strive to give comprehensive answers, especially when asked for explanations, examples, or advice. If appropriate, structure your response with clear sections or points. Include depth of reasoning and examples when relevant. Avoid overly brief answers."}
few_shot_examples = [
    {
        "role": "user", 
        "content": "How do I bake a cake?"
    },
    {
        "role": "assistant", 
        "content": "To bake a cake, follow these steps for a simple vanilla cake:\n\n1. **Ingredients**: You will need 2 cups of flour, 1.5 cups of sugar, 4 eggs, 1 cup of milk, 1 cup of butter, 2 teaspoons of baking powder, and 2 teaspoons of vanilla extract.\n\n2. **Preparation**: Preheat your oven to 350°F (175°C). Grease and flour a 9x13 inch pan. In a medium bowl, mix the flour and baking powder. In a separate bowl, cream the butter and sugar until light and fluffy, then beat in the eggs one at a time, followed by the vanilla extract.\n\n3. **Mixing**: Gradually add the flour mixture to the butter mixture, alternating with the milk. Continue mixing until the batter is smooth.\n\n4. **Baking**: Pour the batter into the prepared pan. Bake for 30-40 minutes or until a toothpick inserted into the center comes out clean. Let the cake cool in the pan for 10 minutes before transferring to a wire rack to cool completely.\n\n5. **Optional**: Once cooled, you can frost the cake with your favorite icing or leave it plain for a simple dessert."
    },
    {
        "role": "user", 
        "content": "What are some ways to improve productivity?"
    },
    {
        "role": "assistant", 
        "content": "Improving productivity can be achieved through several key strategies:\n\n1. **Time Management**: One popular method is time-blocking, where you schedule specific tasks for dedicated time slots throughout the day. This helps you focus on one task at a time without getting overwhelmed by multitasking.\n\n2. **Prioritize Tasks**: Using techniques like the Eisenhower Matrix, you can sort tasks into categories based on their urgency and importance. Focus on tasks that are both urgent and important first, while delegating or postponing less critical tasks.\n\n3. **Use Productivity Tools**: Applications like Trello or Todoist can help organize tasks, set deadlines, and track your progress. These tools provide visual representations of your work and can help with planning.\n\n4. **Break Tasks into Smaller Chunks**: Large tasks can feel overwhelming, but by breaking them down into smaller, more manageable steps, you can maintain momentum and make steady progress.\n\n5. **Take Regular Breaks**: The Pomodoro Technique, which involves working for 25 minutes and then taking a 5-minute break, is a great way to keep focused while avoiding burnout."
    }]


messages = [system_message] + few_shot_examples + [user_message]
formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
```
Cool, now we got amazingly formatted answers from the LLM.

![](/fewShotPrompt.png)


#### Reading List :

[1] https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p <br>
[2] https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py <br>






