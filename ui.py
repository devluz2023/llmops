import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://app:8050").rstrip("/")

st.set_page_config(page_title="Assistente local", page_icon="💬")
st.title("Assistente de IA")
st.caption("Conversa local com Ollama")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou seu assistente local. Como posso ajudar?",
        }
    ]

if st.button("Nova conversa"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou seu assistente local. Como posso ajudar?",
        }
    ]
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escreva sua mensagem"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = httpx.post(
                    f"{API_URL}/chat",
                    json={"message": prompt},
                    headers={"X-Session-ID": st.session_state.session_id},
                    timeout=180.0,
                )
                response.raise_for_status()
                answer = response.json()["response"]
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except ValueError:
                    detail = str(exc)
                st.error(f"Falha ao gerar resposta: {detail}")
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                st.error(f"Não foi possível obter resposta da API: {exc}")
            else:
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

st.divider()
st.subheader("Enviar mensagem pelo WhatsApp")
st.caption("Informe o número no formato internacional, por exemplo +5511999999999.")

with st.form("whatsapp_send_form"):
    whatsapp_phone = st.text_input("Número do WhatsApp")
    whatsapp_message = st.text_area("Mensagem", max_chars=1_600)
    send_whatsapp = st.form_submit_button("Enviar pelo WhatsApp")

if send_whatsapp:
    try:
        response = httpx.post(
            f"{API_URL}/whatsapp/send",
            json={"phone": whatsapp_phone.strip(), "message": whatsapp_message},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except ValueError:
            detail = str(exc)
        st.error(f"Falha ao enviar pelo WhatsApp: {detail}")
    except httpx.HTTPError as exc:
        st.error(f"Não foi possível conectar à API: {exc}")
    else:
        st.success(f"Mensagem enviada para {whatsapp_phone.strip()} pelo WhatsApp.")
