---
title: "Game playing AI."
description: "Exploring the famous minimax algorithm."
pubDate: "Sep 04 2023"
heroImage: "/DeepBlue.webp"
---


Let's approach Artificial Intelligence from the very foundation, shall we? 

We'll build a program - an agent and add attributes to the agent that makes it intelligent - step by step. We'll take the game of Pente. Check this [blog](https://pente.org/help/helpWindow.jsp?file=playGameRules) for rules.

Why choose gaming scenarios for AI? The correlation between AI's involvement in gaming and AI research is reminiscent of the symbiotic relationship between the automotive industry and Formula 1 racing. Many innovations observed in the automotive sector trace their origins back to the competitive world of F1 racing. Similarly, achieving victories over humans in games devised by humans proved to be a brilliant avenue for showcasing the computational prowess underpinning AI research, catalyzing numerous innovations and groundbreaking developments.


Okay, so we need something that effectively uses its compute power to look up information in the future. The process should be deterministic and admissible. Sounds simple right? As an agent, I should make an acceptable, possible move, and with the same policy evaluate all possible moves that the opponent could make and on and on and on - till I reach a condition where either I win or the opponent wins. All this for a single move I make. There can be multiple moves I could make at a stage. All of this would lead to a search space whose navigational complexity would be of O(b^m) complexity, where b is the branching factor and m is the depth till the termination condition. Well, the branching factor b is no small value. It could be as high as 100 and the depth could be a very high value as well. Humanity could cease to exist by the time my agent coughs up a possible move. I need to make the agent better. The rest of the blog will outline how.


Before I get into the optimizations, I'll introduce some terms that I'd like to use in the coming sections. Terminal state is a board configuration when the game is done. Utility function is a function that rewards a terminal state based on the outcome of the game - kinda like +10 for a win, -10 for a loss and 0 for a draw. 
By depth, I mean the recursive depth to which the game goes for a particular move - *you move, I move* kinda thing. 

First, let us try to minimize the branching factor. This problem is classic DFS. When the search space gets insanely big or when the tree gets insanely messy/big, what do you do? Cut off or prune some branches, right? But we need to be careful while cutting off a branch - it needs to be correct - we cannot prune branches that are essential for gameplay. Let us approach the cutting branches thing conceptually.

## Prune 'em branches.

![Minimax gameplay](/minimax.drawio.png)

By principles of recursion and dfs, by the time I'm at C, I'd have visited the nodes A,B,D,E,F,G. I know for a fact that A level is a min-level and the B,C level is a max level. Meaning that A would have a value that is the minimum value of all children of A. So when I'm at node B, I know for a fact that I don't need to explore the branches from B that are greater than A - because these nodes, even though they are useful now, they'll get shadowed by A in the next level min and there is no point in doing any utility state depth recursion from this point on. I can simply break and not be worried about the correctness.

Let's look at an example : 

![Minimax gameplay](/minimax-eg.drawio.png)

Great, we are back at the node C. Now we check the left most child of C. At this point, we have to explore all the possibilities of the node with value 1. Moving on to the next child. At this node, we see that it has a reward value of 11. I know for a fact that if 11 gets picked at my level, there is no way this node is going to be propagated to the root, because at the min level, the minimum value is picked and already from B - 9 is getting picked which is lesser than 11. Exploring other branches in this subtree will also be useless. Why? So there can be three possibilities for values in the rest of the nodes. It could be (a) less than 11 (b) equal to 11 (c) greater than 11. Now in the (a) option, 11 will be picked at the max level; with option (b), since it is equal, it doesn't make a difference; with option (c) when it is greater than, it is definitely not going to be picked in the min level. So we can terminate here itself! Pretty neat right? This is just a simple example where the nodes have values. In a real game playing scenario, these nodes represent a move - a board configuration. Now those are computationally expensive! We really saved up a lot of useless computation with that move. 

What we just did is called the Beta pruning at the Max level. Beta value when we reach C is the value at B. It is minimized in a min level. Similarly for the min level, we prune using a variable called alpha that is maximized in the max level. Together this algorithm is called Alpha beta pruning. Now think about it, the ordering of the nodes can play an amazing speed up in this algorithm. The earlier I encounter 11, the more branches, I can mercilessly cut off. So when the alpha beta pruning is added to the minimax algorithm with efficient ordering, I can cut the branching factor upto sqrt(b), freeing up computation power for me to go twice the depth. But what do I mean by go more depth - we need to go the whole way right? Whole way till either I win or the opponent wins? That is the next section of the blog.


## Can I approximate?

Effectively till this point, what we have done is basically treat the agent's problem as a search problem. We explore the possibilities - we branch out and go with an explorer's attitude : "**Where's my terminal state!?**".
If we go search, we are almost always tempted by the concept of heuristics : can I approximate from this vantage point? Admissible approximate heuristics are a beautiful concept. 
Remember, we are in the AI era. The smartest people of the generation worked on figuring out mathematical admissible heuristics to speeden up our search. Let's rely on them from the game perspective shall we?
So given a board state, I should assign a reward that best describes my chances of winning from that point on. Let's call this an evaluation function. Some board states are really strategic and we should tip our agent in making those moves. These moves will guarentee a win for us. Let us take the double threat as an instance,as described in the Pente forum :

<img src="/Pente.png" width="400" height="400" alt="Pente double threat"/>

There is so much going on that board that there are two things I want to do: I want to be in that state and I never want to be the opponent in that stage. I need to maximize the reward for me getting to that state and minimize the chances of the opponent getting there. So we encode the board information and check for these patterns after every move of the player and that of the opponent. Let us start by checking simple pattern of moves : Capturing a fifth pair, Placing 5 in a row, creating an open 4, creating an open 3 and so on. These rewards should be weighted according to their importance. For instance if I have the possibility of creating an open 4 - I should definitely not be creating another open 3 - I should go in for the kill. So that's why weights.

First off, we need to muster up the code for detecting these patterns and the next thing is to decide on the weights. The first one is simple. Let's see how the second one can be done.

This is a snippet of code from my [github repository](https://github.com/bhargav191098/Pente).

    int w1 = 80000; //Weight for 5 captured or 5 in a row.
    int w2 = 9000; // For creating a open 4.
    int w4 = 400; // For creating a open 3.
    int w3 = (iCapturePoints - theyCapturePoints); //For Capture.
    
    int prod1 = w1*all5s+(depth*-1);
    int prod2 = w1*iCaptured5+(depth*-1);
    int prod3 = w2*opens4+(depth*-1);
    int prod4 = w4*opens3+(depth*-1);
    w3 = w3+(depth*-1);

    if(isMyMoveEvaluated){
        evalScore += (prod1 + prod2+ prod3+prod4+w3);
    }
    else{
        evalScore -= (prod1 + prod2 + prod3 +prod4+w3) ;
    }


The additional term of (depth*-1) incentivizes that I make a decision earlier. If it is possible to arrive at a state earlier I should make that move. Now my agent can strategize neatly. 

Another trick up the sleeve here. Notice in w3, it is difference between iCapturePoints - theyCapturePoints. Which means 5-4 and 3-2 have the same weight. But in reality, a fifth capture should be weighted more than a third capture. This issue can be fixed by taking a scale that grows exponentially. Power it up! 

        iCapturePoints = pow(5,iCapturedPairs);
        theyCapturePoints = pow(5,theyCapturedPairs);

So there we go, we have an agent that (a) makes clever choices (b) has admissible heuristic. Can we now christen it an AI agent? ;)


This homework was done as a part of the CSCI-561: Foundations of AI curriculum taught by Prof. Laurent Itti! Took this course in Spring '23 at USC with some amazing people! Shoutout to Cibi,Sriki and Ananya for being amazing study buddies! I had a lot to learn from the curriculum as well as these people! 