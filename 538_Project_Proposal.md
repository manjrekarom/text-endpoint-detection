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

### 4. On what data will you run your system? (Sriram)

### 5. How will you evaluate the performance of your system? (Omkar + Anh + Sriram)

### 6. What NLP-related difficulties and challenges do you anticipate?

