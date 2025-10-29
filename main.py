from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """Brandon Winn Sanderson (born December 19, 1975) is an American author of high fantasy, science fiction, and young adult books. He is best known for the Cosmere fictional universe, in which most of his fantasy novels, most notably the Mistborn series and The Stormlight Archive, are set. Outside of the Cosmere, he has written several young adult and juvenile series including The Reckoners, the Skyward series,[a] and the Alcatraz series. He is also known for finishing author Robert Jordan's high fantasy series The Wheel of Time. Sanderson has created two graphic novels, including White Sand and Dark One.
Sanderson created Sanderson's Laws of Magic and popularized the idea of "hard magic" and "soft magic" systems. In 2008, Sanderson started a podcast with the horror writer Dan Wells and the cartoonist Howard Tayler called Writing Excuses, involving topics about creating genre writing and webcomics. In 2016, the American media company DMG Entertainment licensed the film rights to Sanderson's entire Cosmere universe, but the rights have since reverted back to Sanderson. Sanderson's March 2022 Kickstarter campaign became the most successful in history, finishing with 185,341 backers pledging US$41,754,153.[3] In mid-2022, Sanderson and Dan Wells started another podcast, Intentionally Blank, which is focused on writing and pop culture.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short Summary
    2. two interesting facts about them"""

    summary_prompt_template = PromptTemplate(
            input_valiables = ["information"], template = summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5", verbose=False)
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    # llm = ChatOllama(temperature=0, model="gpt-oss:latest")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.text)
    
if __name__ == "__main__":
    main()
