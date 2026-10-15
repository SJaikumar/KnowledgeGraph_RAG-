import streamlit as st
from pathlib import Path
from pipeline import DATA, answer

st.set_page_config(page_title='Tamil Nadu Scheme Assistant',page_icon='🌾',layout='wide')
st.title('🌾 Tamil Nadu Schemes Assistant')
st.caption('English / தமிழ் • Source-grounded answers • Check official notifications for current eligibility.')
if not (DATA/'faiss.index').exists():
    st.warning('Knowledge index has not been created. Run: python pipeline.py crawl, then extract, then build.')
    st.stop()
if 'messages' not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Supporting sources'):
                for i,r in enumerate(message['sources'],1):
                    st.markdown(f"[{i}] [{r['name']}]({r['url']})")
question=st.chat_input('Ask about eligibility, subsidy, farmer training... / தமிழில் கேளுங்கள்')
if question:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):st.markdown(question)
    with st.chat_message('assistant'):
        try:
            response,sources=answer(question)
            st.markdown(response)
            with st.expander('Supporting sources'):
                for i,r in enumerate(sources,1):
                    st.markdown(f"[{i}] [{r['name']}]({r['url']})")
            st.session_state.messages.append({'role':'assistant','content':response,'sources':sources})
        except Exception as e:
            st.error(f'Unable to answer: {e}')
