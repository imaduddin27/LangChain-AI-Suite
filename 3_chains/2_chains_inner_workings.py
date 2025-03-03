from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_groq import ChatGroq 

load_dotenv()

llm = llm = ChatGroq(model="llama-3.2-90b-vision-preview")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a facts expert who knows facts about the {animal}"),
        ("human", "Tell me {facts_count} facts.")
    ]
)

# Create individual runnables 
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequemce
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"animal": "cat", "facts_count": 2})

print(response)
