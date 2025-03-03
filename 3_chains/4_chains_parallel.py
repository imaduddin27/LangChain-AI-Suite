from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq 
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel

load_dotenv()

llm = llm = ChatGroq(model="llama-3.2-90b-vision-preview")

# Define prompt template for movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a movie critic."),
        ("human", "provide a brief summary of the movie {movie_name}"),
    ]
)

# Define plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

# Define character analysis step
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
        ]
    )
    return character_template.format_prompt(characters=characters)

def combine_verdicts(plot_analysis, Character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\n Character Analysis:\n{Character_analysis}"

plot_branch_chain = (RunnableLambda(lambda x: analyze_plot(x)) | llm | StrOutputParser())

character_branch_chain = (RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser())


chain = (summary_template | llm | StrOutputParser() | RunnableParallel(branches = {"plot": plot_branch_chain, "characters": character_branch_chain})
        | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"])))

response = chain.invoke({"movie_name": "Inception"})

print(response)