from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import streamlit as st


def get_llm_response(form_input, email_sender, email_recipient, email_style):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    # Template for building the PROMPT
    template = """
    Write a email with {style} style and includes topic :{email_topic}.\n\nSender: {sender}\nRecipient: {recipient}
    \n\nEmail Text:
    """

    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["style", "email_topic", "sender", "recipient"],
        template=template, )

    # Generating the response using LLM
    response = llm.invoke(
        prompt.format(email_topic=form_input, sender=email_sender, recipient=email_recipient, style=email_style))
    # print(response)
    return response


def main():
    st.set_page_config(page_title="Generate Emails",
                       layout='centered',
                       initial_sidebar_state='collapsed')
    st.header("Generate Emails @@@@")
    email_topic = st.text_area('Enter the email topic', height=275)

    # Creating columns for the UI - To receive inputs from user
    col1, col2, col3 = st.columns([10, 10, 5])
    with col1:
        email_sender = st.text_input('Sender Name')
    with col2:
        email_recipient = st.text_input('Recipient Name')
    with col3:
        email_style = st.selectbox('Writing Style',
                                   ('Formal', 'Appreciating', 'Not Satisfied', 'Neutral'),
                                   index=0)
    submit = st.button("Generate")
    if submit:
        email = get_llm_response(
            form_input=email_topic,
            email_sender=email_sender,
            email_recipient=email_recipient,
            email_style=email_style
        )
        st.write(email)


if __name__ == '__main__':
    main()
