import gradio as gr
from openai import OpenAI
import requests
import json
# from volcenginesdkarkruntime import Ark
import torch
import torchaudio
from einops import rearrange
import argparse
import json
import os
from tqdm import tqdm
import random
import numpy as np
import sys
from huggface_diffrhythm.space.DiffRhythm.diffrhythm.infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_style_prompt,
    prepare_model,
    get_negative_style_prompt
)
from diffrhythm.infer.infer import inference

device='cpu'
cfm, tokenizer, muq, vae = prepare_model(device)
cfm = torch.compile(cfm)

def infer_music(lrc, ref_audio_path, steps, sway_sampling_coef_bool, max_frames=2048, device='cpu'):

    sway_sampling_coef = -1 if sway_sampling_coef_bool else None
    lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
    style_prompt = get_style_prompt(muq, ref_audio_path)
    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt = get_reference_latent(device, max_frames)
    generated_song = inference(cfm_model=cfm, 
                               vae_model=vae, 
                               cond=latent_prompt, 
                               text=lrc_prompt, 
                               duration=max_frames, 
                               style_prompt=style_prompt,
                               negative_style_prompt=negative_style_prompt,
                               steps=steps,
                               sway_sampling_coef=sway_sampling_coef,
                               start_time=start_time
                               )
    return generated_song

def R1_infer1(theme, tags_gen, language):
    try:
        client = OpenAI(api_key="XXXX", base_url = "https://ark.cn-beijing.volces.com/api/v3")

        llm_prompt = """
        请围绕"{theme}"主题生成一首符合"{tags}"风格的完整歌词。生成的{language}语言的歌词。
        ### **歌曲结构要求**
        1. 歌词应富有变化，使情绪递进，整体连贯有层次感。**每行歌词长度应自然变化**，切勿长度一致，导致很格式化。
        2. **时间戳分配应根据歌曲的标签\歌词的情感、节奏来合理推测**，而非机械地按照歌词长度分配。 
        ### **歌曲内容要求**
        1. **第一句歌词的时间戳应考虑前奏长度**，避免歌词从 `[00:00.00]` 直接开始。
        2. **严格按照 LRC 格式输出歌词**，每行格式为 `[mm:ss.xx]歌词内容`。
        3. 输出的歌词不能有空行、括号，不能有其他解释内容，例如：副歌、桥段、结尾。  
        4. 输出必须是**纯净的 LRC**。
        """

        response = client.chat.completions.create(
            model="ep-20250215195652-lrff7",
            messages=[
                {"role": "system", "content": "You are a professional musician who has been invited to make music-related comments."},
                {"role": "user", "content": llm_prompt.format(theme=theme, tags=tags_gen, language=language)},
            ],
            stream=False
        )
        
        info = response.choices[0].message.content

        return info

    except requests.exceptions.RequestException as e:
        print(f'请求出错: {e}')
        return {}



def R1_infer2(tags_lyrics, lyrics_input):
    client = OpenAI(api_key="XXX", base_url = "https://ark.cn-beijing.volces.com/api/v3")

    llm_prompt = """
    {lyrics_input}这是一首歌的歌词,每一行是一句歌词,{tags_lyrics}是我希望这首歌的风格，我现在想要给这首歌的每一句歌词打时间戳得到LRC，我希望时间戳分配应根据歌曲的标签、歌词的情感、节奏来合理推测，而非机械地按照歌词长度分配。第一句歌词的时间戳应考虑前奏长度，避免歌词从 `[00:00.00]` 直接开始。严格按照 LRC 格式输出歌词，每行格式为 `[mm:ss.xx]歌词内容`。最后的结果只输出LRC,不需要其他的解释。
    """

    response = client.chat.completions.create(
        model="ep-20250215195652-lrff7",
        messages=[
            {"role": "system", "content": "You are a professional musician who has been invited to make music-related comments."},
            {"role": "user", "content": llm_prompt.format(lyrics_input=lyrics_input, tags_lyrics=tags_lyrics)},
        ],
        stream=False
    )

    info = response.choices[0].message.content

    return info

css = """
/* 固定文本域高度并强制滚动条 */
.lyrics-scroll-box textarea {
    height: 300px !important;  /* 固定高度 */
    max-height: 500px !important;  /* 最大高度 */
    overflow-y: auto !important;  /* 垂直滚动 */
    white-space: pre-wrap;  /* 保留换行 */
    line-height: 1.5;  /* 行高优化 */
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# DiffRhythm")
    
    with gr.Tabs() as tabs:
        
        # page 1
        with gr.Tab("Music Generate", id=0):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Best Practices Guide", open=False):
                        gr.Markdown("""
                        1. **Lyrics Format Requirements**
                        - Each line must follow: `[mm:ss.xx]Lyric content`
                        - Example of valid format:
                            ``` 
                            [00:07.23]Fight me fight me fight me
                            [00:08.73]You made me so unlike me
                            ```

                        2. **Generation Duration Limits**
                        - Current version supports maximum **95 seconds** of music generation
                        - Total timestamps should not exceed 01:35.00 (95 seconds)

                        3. **Audio Prompt Requirements**
                        - Reference audio should be ≥10 seconds for optimal results
                        - Shorter clips may lead to incoherent generation
                        """)
                    lrc = gr.Textbox(
                        label="Lrc",
                        placeholder="Input the full lyrics",
                        lines=12,
                        max_lines=50,
                        elem_classes="lyrics-scroll-box"
                    )
                    audio_prompt = gr.Audio(label="Audio Prompt", type="filepath")
                    
                with gr.Column():
                    steps = gr.Slider(
                                    minimum=10,
                                    maximum=40,
                                    value=32, 
                                    step=1,
                                    label="Diffusion Steps",
                                    interactive=True,
                                    elem_id="step_slider"
                                )
                    sway_sampling_coef_bool = gr.Radio(
                                    choices=[("False", False), ("True", True)],
                                    label="Use sway_sampling_coef",
                                    value=False, 
                                    interactive=True,
                                    elem_classes="horizontal-radio"
                                )
                    lyrics_btn = gr.Button("Submit", variant="primary")
                    audio_output = gr.Audio(label="Audio Result", type="filepath", elem_id="audio_output")
                    
            
            gr.Examples(
                examples=[
                    ["./gift_of_the_world.wav"], 
                    ["./most_beautiful_expectation.wav"],
                    ["./ltwyl.wav"]
                ],
                inputs=[audio_prompt],  
                label="Audio Examples",
                examples_per_page=3
            )

            gr.Examples(
                examples=[
                    ["""[00:10.00]Moonlight spills through broken blinds
[00:13.20]Your shadow dances on the dashboard shrine
[00:16.85]Neon ghosts in gasoline rain
[00:20.40]I hear your laughter down the midnight train
[00:24.15]Static whispers through frayed wires
[00:27.65]Guitar strings hum our cathedral choirs
[00:31.30]Flicker screens show reruns of June
[00:34.90]I'm drowning in this mercury lagoon
[00:38.55]Electric veins pulse through concrete skies
[00:42.10]Your name echoes in the hollow where my heartbeat lies
[00:45.75]We're satellites trapped in parallel light
[00:49.25]Burning through the atmosphere of endless night
[01:00.00]Dusty vinyl spins reverse
[01:03.45]Our polaroid timeline bleeds through the verse
[01:07.10]Telescope aimed at dead stars
[01:10.65]Still tracing constellations through prison bars
[01:14.30]Electric veins pulse through concrete skies
[01:17.85]Your name echoes in the hollow where my heartbeat lies
[01:21.50]We're satellites trapped in parallel light
[01:25.05]Burning through the atmosphere of endless night
[02:10.00]Clockwork gears grind moonbeams to rust
[02:13.50]Our fingerprint smudged by interstellar dust
[02:17.15]Velvet thunder rolls through my veins
[02:20.70]Chasing phantom trains through solar plane
[02:24.35]Electric veins pulse through concrete skies
[02:27.90]Your name echoes in the hollow where my heartbeat lies"""],
                ["""[00:05.00]Stardust whispers in your eyes
[00:09.30]Moonlight paints our silhouettes
[00:13.75]Tides bring secrets from the deep
[00:18.20]Where forever's breath is kept
[00:22.90]We dance through constellations' maze
[00:27.15]Footprints melt in cosmic waves
[00:31.65]Horizons hum our silent vow
[00:36.10]Time unravels here and now
[00:40.85]Eternal embers in the night oh oh oh
[00:45.25]Healing scars with liquid light
[00:49.70]Galaxies write our refrain
[00:54.15]Love reborn in endless rain
[01:15.30]Paper boats of memories
[01:19.75]Float through veins of ancient trees
[01:24.20]Your laughter spins aurora threads
[01:28.65]Weaving dawn through featherbed"""]
                ],
                inputs=[lrc], 
                label="Lrc Examples",
                examples_per_page=2
            )
  
        # page 2
        with gr.Tab("LLM Generate LRC", id=1):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Notice", open=False):
                        gr.Markdown("**Two Generation Modes:**\n1. Generate from theme & tags\n2. Add timestamps to existing lyrics")
                    
                    with gr.Group():
                        gr.Markdown("### Method 1: Generate from Theme")
                        theme = gr.Textbox(label="theme", placeholder="Enter song theme, e.g. Love and Heartbreak")
                        tags_gen = gr.Textbox(label="tags", placeholder="Example: male pop confidence healing")
                        language = gr.Dropdown(["zh", "en"], label="language", value="en")
                        gen_from_theme_btn = gr.Button("Generate LRC (From Theme)", variant="primary")

                    with gr.Group(visible=True): 
                        gr.Markdown("### Method 2: Add Timestamps to Lyrics")
                        tags_lyrics = gr.Textbox(label="tags", placeholder="Example: female ballad piano slow")
                        lyrics_input = gr.Textbox(
                            label="Raw Lyrics (without timestamps)",
                            placeholder="Enter plain lyrics (without timestamps), e.g.:\nYesterday\nAll my troubles...",
                            lines=12,
                            max_lines=50,
                            elem_classes="lyrics-scroll-box"
                        )
                        gen_from_lyrics_btn = gr.Button("Generate LRC (From Lyrics)", variant="primary")

                with gr.Column():
                    lrc_output = gr.Textbox(
                        label="Generated LRC Lyrics",
                        placeholder="Timed lyrics will appear here",
                        lines=50,
                        elem_classes="lrc-output",
                        show_copy_button=True
                    )
                    
            # Examples section
            gr.Examples(
                examples=[
                    [
                        "Love and Heartbreak", 
                        "female vocal emotional piano pop",
                        "en"
                    ],
                    [
                        "Heroic Epic", 
                        "male choir orchestral powerful",
                        "zh"
                    ]
                ],
                inputs=[theme, tags_gen, language],
                label="Examples: Generate from Theme"
            )

            gr.Examples(
                examples=[
                    [
                        "acoustic folk happy", 
                        """I'm sitting here in the boring room
                        It's just another rainy Sunday afternoon"""
                    ],
                    [
                        "electronic dance energetic",
                        """We're living in a material world
                        And I am a material girl"""
                    ]
                ],
                inputs=[tags_lyrics, lyrics_input],
                label="Examples: Generate from Lyrics"
            )

            # Bind functions
            gen_from_theme_btn.click(
                fn=R1_infer1,
                inputs=[theme, tags_gen, language],
                outputs=lrc_output
            )
            
            gen_from_lyrics_btn.click(
                fn=R1_infer2,
                inputs=[tags_lyrics, lyrics_input],
                outputs=lrc_output
            )

    tabs.select(
    lambda s: None, 
    None, 
    None 
    )
    
    lyrics_btn.click(
        fn=infer_music,
        inputs=[lrc, audio_prompt, steps, sway_sampling_coef_bool],
        outputs=audio_output
    )
    
demo.queue().launch(show_api=False, show_error=True)



if __name__ == "__main__":
    demo.launch()