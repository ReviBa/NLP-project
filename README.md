# Chat with a character from a TV-series
This code base suggest 2 methods for persona-based conversation modeling and conduct comparison strategies between them. 
Our main goal was to build a chatbot that imitates a character from a TV-series. 

For this task we experimented with 2 different models and compared their results. 
The first one is more trivial and includes fine-tuning a dialog model on the transcript. 
However, the second one is more innovative and related to a style transfer task. 
Both methods involved the fine-tuning of a T5 dialog - a versatile language model renowned for its text-to-text framework

The dataset used for these models is extracted from the complete dialogue transcript of the TV-series ”The Office,” with a focus on the character Michael Scott, who occupies the central role.

## Code Structure
For each model, we present 3 main resources:
1. Dataset creation and processing (under data_pre_processing/)
2. We save the resuted dataset under resources/
3. Training (under training/)
4. The evaluation of the models is under the corresponding folder

# Usage
We created a playground.ipynb file for you, where you can load the models and interact with them.
here some example of our models unswers: (TODO)
