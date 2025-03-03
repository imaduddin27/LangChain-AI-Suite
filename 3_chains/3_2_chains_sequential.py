from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain_groq import ChatGroq 

load_dotenv()

llm = ChatGroq(model="llama-3.2-90b-vision-preview")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a facts expert who knows facts about the {animal}"),
        ("human", "Tell me {facts_count} facts.")
    ]
)


translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a translator and convert the provided text into {language}"),
        ("human", "Translate the following text to {language}: {text}")
    ]
)

chain = (
    animal_facts_template | llm | StrOutputParser() | (lambda output: {"text": output, "language": "{russian}"})
    | translation_template | llm | StrOutputParser()
)

result = chain.invoke({"animal": "cat", "facts_count": 2})

print(result)
