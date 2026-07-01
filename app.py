import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
import tempfile
import os

# =========================
# CONFIG
# =========================

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.set_page_config(
    page_title="AI Announcer Robot",
    layout="centered"
)

st.title("🤖 AI Announcer Robot")

st.info(
    "Enter a message and convert it into a professional broadcast-style announcement."
)

# =========================
# INPUT
# =========================

text = st.text_input(
    "Enter announcement message"
)

# =========================
# GENERATE ANNOUNCEMENT
# =========================

if st.button("Generate Announcement"):

    if not text.strip():

        st.warning(
            "Please enter an announcement."
        )

    elif not HF_TOKEN:

        st.error(
            "HF_TOKEN is missing."
        )

    else:

        with st.spinner(
            "Generating announcement..."
        ):

            try:

                prompt = f"""
You are a professional public announcement system.

Convert the following message into a formal announcement.

Message:
{text}

Requirements:
- Professional tone
- Clear and concise
- Suitable for schools, offices, airports, or events
- Begin with an attention phrase if appropriate
"""

                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=250,
                    temperature=0.5
                )

                announcement = (
                    response.choices[0]
                    .message
                    .content
                )

                st.subheader(
                    "📢 Announcement"
                )

                st.write(
                    announcement
                )

                # =========================
                # TEXT TO SPEECH
                # =========================

                try:

                    tts = gTTS(
                        text=announcement
                    )

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp3"
                    ) as fp:

                        tts.save(fp.name)

                        st.subheader(
                            "🔊 Announcement Audio"
                        )

                        st.audio(fp.name)

                except Exception as e:

                    st.error(
                        f"TTS Error: {e}"
                    )

            except Exception as e:

                st.error(
                    f"System Response: {e}"
                )
