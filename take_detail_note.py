from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import streamlit as st


def get_llm_response(user_input_text, notes_style):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    # Template for building the PROMPT
    template = """
    Take notes from below text, highlighting the key ideas, breaking down each section, 
    and summarizing supporting examples or arguments. 
    Organize the notes in bullet points, and use headings for major sections with style {notes_style}.
    Don't make up idea, based on the text only.
    \nUser Input: {user_input_text}
    """

    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["user_input_text", "notes_style"],
        template=template, )

    # Generating the response using LLM
    response = llm.invoke(
        prompt.format(user_input_text=user_input_text,
                      notes_style=notes_style))
    # print(response)
    return response


def main():
    st.set_page_config(page_title="Take notes assistance",
                       layout='centered',
                       initial_sidebar_state='collapsed')
    st.header("Key Notes!")
    user_input_text = st.text_area('Enter the text: ', height=275)

   
    notes_style = st.selectbox('Detail',
                                   ('Detail', 'General'),
                                   index=0)
    submit = st.button("Generate")
    if submit:
        st.write("In progress ...")
       
        email = get_llm_response(
            user_input_text=user_input_text,
            notes_style=notes_style
        )
        st.write(email)


if __name__ == '__main__':
    main()
