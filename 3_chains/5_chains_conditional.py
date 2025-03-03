from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq 
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch

load_dotenv()

llm = llm = ChatGroq(model="llama-3.2-90b-vision-preview")

# define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant"),
        ("human", "Generate a thank you note for this positive feedback: {feedback}. Just a short note. not an email.",),
    ]
)

negative_feedback_template = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}"),
    ]
)

neutral_feedback_template = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant"),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}"),
    ]
)

escalate_feedback_template = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant"),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}"),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "ou are a helpful assitant"),
        ("human", "Classify the statement of this feedback as positive, negative, neutral or escalate: {feedback}"),
    ]
)

# Define the runnable branches for appropriate feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | llm | StrOutputParser()  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | llm | StrOutputParser()  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | llm | StrOutputParser()  # Neutral feedback chain
    ),
    escalate_feedback_template | llm | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template | llm | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches 

# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "I would like to talk with the customer support human agent."
result = chain.invoke({"feedback": review})

# Output the result
print(result)