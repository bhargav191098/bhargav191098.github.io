---
title: "Man, metal and models - 1: The plan"
description: "Learning LLM Inference."
pubDate: "Sept 28 2024"
heroImage: "/llama.jpg"
---

I got to admit. Building the software around large models is fun. The scale of both the model and the users that are using these models is crazy.

It feels like in the past 5 years, I've always tried to understand Deep Learning in my own ways: applying these models, trying out clever ways of transfer learning on them to make the models converge faster. But to be very honest, I am an absolute outsider to the algorithimic optimizations of Deep Learning, in the sense that I can understand how these optimizations work but I don't think I'll ever have an <b>original idea</b> that might revolutionize the way we solve a problem. <i>That ship has sailed.</i> Maybe it's because of that, the <b>scaling</b> of Deep Learning looks <i>utterly magical</i> to me. 

For instance, when I was reading the Llama paper, the decision of KV Cache seemed natural to me. But this would mean now that the memory movement operations are the bottleneck. To make up for that, the authors follow that up with the Grouped Multi Query attention as a compromise between speed and accuracy (The age old debate). It feels magical to me that in Deep Learning, even if you compromise on the head being part of Q,K,V and you group it to appease the memory movement, it still works. Llama based models are kinda changing the world as I'm typing this. 

In my own ways, I want to contribute to the software that keeps the scaling possible. This summer, I built a GPU Scheduler that provisions GPUs from cloud providers for deployment and training jobs. That was a good experience in doing system design at scale. Since it's a start-up you have more skin in the game and it's something that I surely would have missed at big tech. I want to build on this and I want to make legitimate contributions to open source platforms in this domain or atleast fail by trying for it. That is one of my goals for this semester.

So I'd simply have a model, metal access thanks to RunPod, and myself. Let's see what happens. Hope this post contains a bunch of follow up articles in which I slowly take up problem statements incrementally and keep solving.

<figure>
<img src="/llmInferenceSD.png">
<figcaption style="text-align:center;font-weight:bold;"> Mind map for learning LLM Inference </figcaption>
</figure>

I want to get to the model and vllm layer by the end of this week. I'm a big fan of how effective the vllm folks are and they have office hours in case I run into some issues. So safest bet.
