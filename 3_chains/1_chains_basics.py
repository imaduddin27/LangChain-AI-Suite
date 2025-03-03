from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq 

load_dotenv()

llm = llm = ChatGroq(model="llama-3.2-90b-vision-preview")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a facts expert who knows facts about the {animal}"),
        ("human", "Tell me {facts_count} facts.")
    ]
)

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"animal": "alligator", "facts_count": 1})

print(result)