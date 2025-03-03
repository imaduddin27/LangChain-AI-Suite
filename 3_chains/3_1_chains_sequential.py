from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_groq import ChatGroq 
from langchain.schema.output_parser import StrOutputParser


load_dotenv()

llm = llm = ChatGroq(model="llama-3.2-90b-vision-preview")

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

# Define additional processing steps using the RunnableLambda
count_words = RunnableLambda(lambda x:f"word count: {len(x.split())}\n{x}" )
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "german"})

chain = animal_facts_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser()

result = chain.invoke({"animal": "cat", "facts_count": 2})

print(result)