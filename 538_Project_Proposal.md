### Possible Titles

1. Should I Reply Now?
2. How long to wait before replying?
3. Goal/Task/Text/Message/Intent Endpoint/Boundary Detection
4. Determining intent across many messages? Possible extension: Replying to messages with multiple intents.
5. 



### 1. What is the problem or task that you propose to solve? (Omkar)

When you are chatting with your friend, you seldom type a whole body of the message and press send. Instead you will type one message after another and your friend knows to respond when you have specified the required information. 

Most task-oriented conversational agents ("chatbots") in use, require user to specify all the information in one message like an email and less like a chat. If the user doesn't specify complete information in one message, the agent either detects the intent incorrectly or detects UNKOWN intent ("I don't know what that means").

Multiple messages provide extra information which might be needed by the receiver to reply. In contrary to email, in a setting like chat, this information can be made available **by-request** by the receiver or **by self-realisation** of the sender. This makes text messaging less formal and more flexible (and is also the reason why people might prefer messaging over email).

The purpose of this project "How long to wait before replying?" is to make conversational agents chat more naturally by attempting to solve this problem. 

**Problem description:**

We call this the task of "ENTER APPROPRIATE TITLE".

A variant of this task appears in speech processing where the sentence boundaries aren't explicit.



### 2. What is interesting about this problem from an NLP perspective? (Omkar) 

The problem requires an agent to understand or learn following things about natural language:

1. Is the text complete "in an informal / spoken way"? The sentence might have grammatical mistakes or no punctuation.
2. Does the text make sense semantically (and syntactically)?
3. Does the text contain **information** required by the receiver to answer? This may be conditioned on the agent's goal (where goal can be pizza delivery, restaurant booking, or just chitchat, etc.).
4. 



### 3. What technical method or approach will you use? (Anh)

* Word Embedding: because our work depends on the meaning of user's utterances, using contextual embedding such as ELMo is a good start. In [Le et al], the author used a combination of GloVe, ELMo, and CNN chars embedding. In my opinion, it's a bit unnecessary and it could increase the training/inference time but we definitely consider it as a potential way to improve out baseline.

* SBD as Classification task (predict the token)

  The task is given a set of utterances, called $S = {s_1, s_2, ..., s_n}$. Then a encoder compresses user's messages into a vector $c$ to use in the classifier layers.

![image](/Users/huyanh/OneDrive - Stony Brook University/Fall 2021/CSE538 Natural Language Processing/Project/nlp-project/model1.jpeg)

​		The objective task is formalized as:

$$P(y|s_1, s_2, ..., s_n) = P(y|x_1, x_2, ..., x_m)$$

​		There are many encoders architecture we could try such as RNN-variant, Transformers, BERT, etc.

* SBD as Sequential Label Task

  In the first approach, we need a trigger to know when the users finished their typing (for example from the time between each utterance) or we have to feed the encoder the new data every time user finished a chat, i.e the data will be ${s_1, (s_1, s_2), (s_1, s_2, s_3),...}$. This could be a trouble if the model has high complexity and takes time to generate a context vector. We should not make users wait. Thus the second approach will take a stream data as an input and generate label sequence with the same length as input. In this approach, the labels set consists of three tokens ${W, ESEN, ECON}$ which denote words, end of sentence and end of context.

  ![image](/Users/huyanh/OneDrive - Stony Brook University/Fall 2021/CSE538 Natural Language Processing/Project/nlp-project/model2.jpeg)
  
  Because formulized as a Neuram Machine Translation task, then the probability of a word $y_i$ is
  
  $$P(y_i|x_1, x_2, ... x_k)$$		

When ever we see the ECON token, we should pass the context vector or sequence of input words to chatbot. There are several studies on the same problem, where we need to generate output sequence simultaneously with the input sequence. One of the work is from [Cho et al] where the authors proposed a new algorithm for decoder named Simultaneously Greedy Decoding. It's complicated where we need two new hyparameters to control delay and quality. For our problem, we will prioritize quality over delay a bit since it's bad to interupt user.



### 4. On what data will you run your system? (Sriram)

### 5. How will you evaluate the performance of your system? (Omkar + Anh + Sriram)

* For the first approach, I think we should use Precision Recall and try to minimize the False Positive. It's bad when the chatbot interupts user being talking.
* The second approach, the authors proposed a novel metrics

### 6. What NLP-related difficulties and challenges do you anticipate?

