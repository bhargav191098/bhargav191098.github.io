---
title: "Man, metal and models - 3: Streaming tokens"
description: "Stream 'em"
pubDate: "Oct 1, 2024"
heroImage: "/gpumeme.png"
---

### The world of token streaming

I think I first saw tokens being streamed in a GPT-2 demo by Ilya. It was not as accurate as the beast we have now, but OpenAI cracked these chat completions as a long time ago. In fact, there is some thing like OpenAPI Compatible server that people can configure their LLM online setup such that they can use the OpenAI library. For instance, let's take a platform I'm a big fan of - TogetherAI :

![together](/openAICompatible.png)

So basically you write the backend in such a way (you ensure that there is an endpoint /v1/chat/completions, the request should have certain body elements and so on.) that you can plug in your endpoint in the place of an OpenAI endpoint.

In this case, you're client side code can make use of the OpenAI streaming completions! Let's bring our friend Together AI again :

![togetherStream](/streamOpenAI.png)

Well oiled machinery right? This is what keeps running in most of the streaming token screens you see. Specifically, I want you to zoom in on the ```chunk.choices[0].delta.content or "", end=""```, this is a liberating piece of code on the client that I struggled to implement in my attempts to break the black box and learn vllm. 

Okay, back to the white board. We want to do the following :

![streamSystemDiagram](/streamSD.jpg)

Before I dig into how to do this, I want to clarify something. vLLM comes in two flavours : offline and online. This is not just for vLLM, but it's true for any LLM inference engine platform that you will come across. Offline is where we get to run the engine, see how the engine works token-level. Online is more like a server - you run a command, it spews up an Open AI compatible server. I would suggest, for people interested on the applications of LLMs, they can simply stick to the online approach. I am an engineer interested in making the conversion from offline -> online. I believe it can unlock a skill for me. Okay storytime over. Back to the problem.

### The beauty of Async Engine:

Okay I think I am ready to start fanboying about vLLM at this point. Here's what I did to understand their platform : Cloned up the vllm-project and digged in a depth-first fashion into the AsyncEngine to understand how it works. The Async Engine is the one that best suited what I wanted to do.

Inference classically works in two steps - <br>
1. Prefill : We basically take the freshly arrived prompt. We do the tokenization process, populate the intermediate state values we need ( remember the key,value from the Llama diagram?) - basically setup the stage for the first token generation. The beauty of the this phase is that you have all the input tokens present, so you can parallelize the shit of this stage and utilize GPU compute for this. <br>
2. Decode : This is the step by step token generation phase - where we incrementally feed the LLM output from itself.<br>

Essentially, every LLM inference engine needs to do this.

The AsyncLLMEngine does this asynchronously. It has a request queue and a scheduler that basically decides at the engine step (smallest unit of execution), what the engine should do. The scheduler sticks to the first come first serve rule as a priority. It does take up requests asynchronously. But if there is memory only for one more request, it will pick a request already in decoding stage, than going for a new request pre-fill. This makes sense from the memory point of view as well. (more about that in later blogs)

Okay, the below image is the magic by which your prompt gets converted to the streaming response you see on your screen.
One token at a time.

![asyncEngineFlow](/async_engine_flow.jpg)

Now, diving into the code, we retain most of the prompt engineering level stuff from the previous post and the fastapi structure.

First step, create the engine object : <br>

```python
engine_args = AsyncEngineArgs(model="meta-llama/Meta-Llama-3.1-8B",max_model_len=max_model_len,disable_async_output_proc=False)
model = AsyncLLMEngine.from_engine_args(engine_args)
```

Oh another teeny tiny detail, in our structured chat few shot example we give, we add <|system|>,<|assistant|> to mark the sections, so when you structure the prompt, add <|assistant|> to the end of the prompt, so the LLM skips generating the marking tags. This is desired because we are streaming. We could do post-processing, but anytime I have to pick up regex, I feel like a scared freshman all over again. 

```python
formatted_prompt += "<|assistant|> :"
```

The end point needs to be changed like this :

```python
@app.post("/generate")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    formatted_prompt = get_formatted_prompt(prompt)
    sampling_params = SamplingParams(
        max_tokens=  max_new_tokens,
        frequency_penalty=0.9,
        temperature= temperature,
        top_p=top_p,
        top_k=top_k,
        stop=["<|system|>","<|user|>"]
    )
    request_id = random_uuid()

    assert model is not None
    results_generator = model.generate(formatted_prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        last_len = 0
        async for request_output in results_generator:
            to_yield = request_output.outputs[-1].text[last_len:]
            print(f"For request_id : ${request_output.request_id} decode step yielded ${to_yield}")
            last_len = len(request_output.outputs[-1].text)
            yield to_yield
            
    
    return StreamingResponse(stream_results())
```

See how I am tracking the last_len to keep sending deltas, something that is available straight out of the box with Open AI chat completions API. Congrats, you have stuck with me till the end in this endevour to reinvent the wheel. xD 

But this streaming backend needs a streaming UI! I am going to make a quick setup in streamlit to make this possible.

```python
import streamlit as st
import requests
import time

st.title("Talk with 3.1")

model_url = "<ENDPOINT_URL>"

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


st.session_state['token_count'] = 0

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def predict():
    payload = {
            "prompt":st.session_state.messages[-1]["content"]
        }
        
    response = requests.post(model_url, json=payload, stream=True)
    partial_message = ""
    for chunk in response.iter_content(chunk_size=20):
        if chunk:
            data = chunk.decode('utf-8')
            yield data


if prompt := st.chat_input("Hey there! Drop any question."):
    #st.session_state.messages.append(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        start_time = time.time()
        response = st.write_stream(predict,)
        st.session_state.messages.append({"role":"assistant","content":response})
        latency_display = st.empty()
        latency_display.markdown(f"**Time latency for server (End to End) :** {time.time() - start_time} seconds.")
```

All set!

![metalMeetsMan](/metalMeetsMan.png)













