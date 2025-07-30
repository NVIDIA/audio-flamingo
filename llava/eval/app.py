import gradio as gr
import torch
import llava
from peft import PeftModel
import os
from huggingface_hub import snapshot_download
import copy 
# ---------------------------------
# SINGLE-TURN MODEL SETUP
# ---------------------------------

MODEL_BASE_SINGLE = snapshot_download(repo_id="nvidia/audio-flamingo-3")
MODEL_BASE_THINK = os.path.join(MODEL_BASE_SINGLE, 'stage35')

# model_single = llava.load(MODEL_BASE_SINGLE, model_base=None, devices=[0])
model_single = llava.load(MODEL_BASE_SINGLE, model_base=None)
model_single = model_single.to("cuda")
model_single_copy = copy.deepcopy(model_single)

generation_config_single = model_single.default_generation_config

model_think = PeftModel.from_pretrained(
        model_single,
        MODEL_BASE_THINK,
        device_map="auto",
        torch_dtype=torch.float16,
        )

# # ---------------------------------
# # MULTI-TURN MODEL SETUP
# # ---------------------------------
# MODEL_BASE_MULTI = snapshot_download(repo_id="nvidia/audio-flamingo-3-chat")
# model_multi = llava.load(MODEL_BASE_MULTI, model_base=None, devices=[0])
# generation_config_multi = model_multi.default_generation_config


# ---------------------------------
# SINGLE-TURN INFERENCE FUNCTION
# ---------------------------------
def single_turn_infer(audio_file, prompt_text):
    try:
        sound = llava.Sound(audio_file)
        full_prompt = f"<sound>\n{prompt_text}"
        response = model_single_copy.generate_content([sound, full_prompt], generation_config=generation_config_single)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# def speech_prompt_infer(audio_prompt_file):
#     try:
#         sound = llava.Sound(audio_prompt_file)
#         full_prompt = "<sound>"
#         response = model_multi.generate_content([sound, full_prompt], generation_config=generation_config_single)
#         return response
#     except Exception as e:
#         return f"❌ Error: {str(e)}"

def think_infer(audio_file, prompt_text):
    try:
        sound = llava.Sound(audio_file)
        full_prompt = f"<sound>\n{prompt_text}"
        response = model_think.generate_content([sound, full_prompt], generation_config=generation_config_single)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------------------------------
# MULTI-TURN INFERENCE FUNCTION
# ---------------------------------
# def multi_turn_chat(user_input, audio_file, history, current_audio):
#     try:
#         if audio_file is not None:
#             current_audio = audio_file  # Update state if a new file is uploaded

#         if current_audio is None:
#             return history + [("System", "❌ Please upload an audio file before chatting.")], history, current_audio

#         sound = llava.Sound(current_audio)
#         prompt = f"<sound>\n{user_input}"

#         response = model_multi.generate_content([sound, prompt], generation_config=generation_config_multi)

#         history.append((user_input, response))
#         return history, history, current_audio
#     except Exception as e:
#         history.append((user_input, f"❌ Error: {str(e)}"))
#         return history, history, current_audio
# ---------------------------------
# INTERFACE
# ---------------------------------
with gr.Blocks(css="""
.gradio-container { 
    max-width: 100% !important; 
    width: 100% !important;
    margin: 0 !important; 
    padding: 0 !important;
}
#component-0, .gr-block.gr-box { 
    width: 100% !important; 
}
.gr-block.gr-box, .gr-column, .gr-row {
    padding: 0 !important;
    margin: 0 !important;
}
""") as demo:

    with gr.Column():
        gr.HTML("""
<div align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/audio-flamingo/audio_flamingo_3/static/logo-no-bg.png" alt="Audio Flamingo 3 Logo" width="120" style="margin-bottom: 10px;">
  <h2><strong>Audio Flamingo 3</strong></h2>
  <p><em>Advancing Audio Intelligence with Fully Open Large Audio-Language Models</em></p>
</div>
<div align="center" style="margin-top: 10px;">
  <a href="https://arxiv.org/abs/2507.08128">
    <img src="https://img.shields.io/badge/arXiv-2503.03983-AD1C18" alt="arXiv" style="display:inline;">
  </a>
  <a href="https://research.nvidia.com/labs/adlr/AF3/">
    <img src="https://img.shields.io/badge/Demo%20page-228B22" alt="Demo Page" style="display:inline;">
  </a>
  <a href="https://github.com/NVIDIA/audio-flamingo">
    <img src="https://img.shields.io/badge/Github-Audio_Flamingo_3-9C276A" alt="GitHub" style="display:inline;">
  </a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers">
    <img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social" alt="GitHub Stars" style="display:inline;">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-3">
    <img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/nvidia/audio-flamingo-3-chat">
    <img src="https://img.shields.io/badge/🤗-Checkpoints_(Chat)-ED5A22.svg">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/datasets/nvidia/AudioSkills">
    <img src="https://img.shields.io/badge/🤗-Dataset:_AudioSkills--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/LongAudio">
    <img src="https://img.shields.io/badge/🤗-Dataset:_LongAudio--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Chat">
    <img src="https://img.shields.io/badge/🤗-Dataset:_AF--Chat-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Think">
    <img src="https://img.shields.io/badge/🤗-Dataset:_AF--Think-ED5A22.svg">
  </a>
</div>
""")
    # gr.Markdown("#### NVIDIA (2025)")

    with gr.Tabs():
        # ---------------- SINGLE-TURN ----------------
        with gr.Tab("🎯 Single-Turn Inference"):
            with gr.Row():
                with gr.Column():
                    audio_input_single = gr.Audio(type="filepath", label="Upload Audio")
                    prompt_input_single = gr.Textbox(label="Prompt", placeholder="Ask a question about the audio...", lines=8)
                    btn_single = gr.Button("Generate Answer")

                    gr.Examples(
                        examples=[
                            ["static/emergent/audio1.wav", "What is surprising about the relationship between the barking and the music?"],
                            ["static/audio/audio2.wav", "Please describe the audio in detail."],
                            ["static/speech/audio3.wav", "Transcribe any speech you hear."],
                        ],
                        inputs=[audio_input_single, prompt_input_single],
                        label="🧪 Try Examples"
                    )

                with gr.Column():
                    output_single = gr.Textbox(label="Model Response", lines=15)

            btn_single.click(fn=single_turn_infer, inputs=[audio_input_single, prompt_input_single], outputs=output_single)
        with gr.Tab("🤔 Think / Long"):

            with gr.Row():
                with gr.Column():
                    audio_input_think = gr.Audio(type="filepath", label="Upload Audio")
                    prompt_input_think = gr.Textbox(label="Prompt", placeholder="To enable thinking, please add the text: '\nPlease think and reason about the input music before you respond.' to your prompt.", lines=8)
                    btn_think = gr.Button("Generate Answer")

                    gr.Examples(
                        examples=[
                            ["static/think/audio1.wav", "What are the two people doing in the audio Choose the correct option from the following options:\n(A) One person is demonstrating how to use the equipment\n(B) The two people are discussing how to use the equipment\n(C) The two people are disassembling the equipment\n(D) One person is teaching another person how to use a piece of equipment\n"],
                            ["static/think/audio2.wav", "Is the boat in the video moving closer or further away? Choose the correct option from the following options:\n(A) Closer\n(B) Further\n"],
                            ["static/speech/videoplayback.wav", "Generate a detailed caption for the input audio, describing all notable speech, sound, and musical events comprehensively. In the caption, transcribe all spoken content by all speakers in the audio precisely."],
                            ["static/speech/speaker1.flac", "Transcribe any input speech in the input audio."],
                        ],
                        inputs=[audio_input_think, prompt_input_think],
                        label="🧪 Try Examples"
                    )

                with gr.Column():
                    output_think = gr.Textbox(label="Model Response", lines=30)

            btn_think.click(fn=think_infer, inputs=[audio_input_think, prompt_input_think], outputs=output_think)
        # ---------------- MULTI-TURN CHAT ----------------
        with gr.Tab("💬 Multi-Turn Chat"):
            # chatbot = gr.Chatbot(label="Audio Chatbot")
            # audio_input_multi = gr.Audio(type="filepath", label="Upload or Replace Audio Context")
            # user_input_multi = gr.Textbox(label="Your message", placeholder="Ask a question about the audio...", lines=8)
            # btn_multi = gr.Button("Send")
            # history_state = gr.State([])           # Chat history
            # current_audio_state = gr.State(None)   # Most recent audio file path

            # btn_multi.click(
            #     fn=multi_turn_chat,
            #     inputs=[user_input_multi, audio_input_multi, history_state, current_audio_state],
            #     outputs=[chatbot, history_state, current_audio_state]
            # )
            # gr.Examples(
            #     examples=[
            #         ["static/chat/audio1.mp3", "This track feels really peaceful and introspective. What elements make it feel so calming and meditative?"],
            #         ["static/chat/audio2.mp3", "Switching gears, this one is super energetic and synthetic. If I wanted to remix the calming folk piece into something closer to this, what would you suggest?"],
            #     ],
            #     inputs=[audio_input_multi, user_input_multi],
            #     label="🧪 Try Examples"
            # )
            # Add the link to another Gradio demo here
            gr.Markdown("🔗 [Check out our other Gradio demo here](https://huggingface.co/spaces/nvidia/audio-flamingo-3-chat)")

        with gr.Tab("🗣️ Speech Prompt"):
            # gr.Markdown("Use your **voice** to talk to the model.")

            # with gr.Row():
            #     with gr.Column():
            #         speech_input = gr.Audio(type="filepath", label="Speak or Upload Audio")
            #         btn_speech = gr.Button("Submit")
            #     gr.Examples(
            #             examples=[
            #                 ["static/voice/voice_0.mp3"],
            #                 ["static/voice/voice_1.mp3"],
            #                 ["static/voice/voice_2.mp3"],
            #             ],
            #             inputs=speech_input,
            #             label="🧪 Try Examples"
            #         )
            #     with gr.Column():
            #         response_box = gr.Textbox(label="Model Response", lines=15)

            # btn_speech.click(fn=speech_prompt_infer, inputs=speech_input, outputs=response_box)
            # Add the link to another Gradio demo here
            gr.Markdown("🔗 [Check out our other Gradio demo here](https://huggingface.co/spaces/nvidia/audio-flamingo-3-chat)")

        # ---------------- ABOUT ----------------
        with gr.Tab("📄 About"):
            gr.Markdown("""
### 📚 Overview
**Audio Flamingo 3** is a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces:
(i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music;
(ii) flexible, on-demand thinking, allowing the model to do chain-of-thought reasoning before answering;
(iii) multi-turn, multi-audio chat;
(iv) long audio understanding and reasoning (including speech) up to 10 minutes; and
(v) voice-to-voice interaction.
To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.
**Key Features:**
💡 Audio Flamingo 3 has strong audio, music and speech understanding capabilities.
💡 Audio Flamingo 3 supports on-demand thinking for chain-of-though reasoning.
💡 Audio Flamingo 3 supports long audio and speech understanding for audios up to 10 minutes.
💡 Audio Flamingo 3 can have multi-turn, multi-audio chat with users under complex context.
💡 Audio Flamingo 3 has voice-to-voice conversation abilities.
""")

    gr.Markdown("© 2025 NVIDIA | Built with ❤️ using Gradio + PyTorch")


# -----------------------
# Launch App
# -----------------------
if __name__ == "__main__":
    demo.launch(share=True)
