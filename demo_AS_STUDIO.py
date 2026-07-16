#!/usr/bin/env python3
import argparse
import logging
import os
import time
import gradio as gr
import numpy as np
import torch
import soundfile as sf
import librosa
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
try:
    from ghx.mp3 import wav_to_mp3
except:
    def wav_to_mp3(p): pass

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": ["American Accent", "Australian Accent", "British Accent"],
    "Chinese Dialect": ["Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect"]
}

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    return parser

def build_demo(model, checkpoint):
    sampling_rate = model.sampling_rate
    def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale, denoise_opt, speed, duration, preprocess_opt, postprocess_opt, remove_silence, mode, ref_text=None):
        if not text or not text.strip(): return None, "Please enter text."
        gen_config = OmniVoiceGenerationConfig(num_step=int(num_step or 32), guidance_scale=float(guidance_scale or 2.0), denoise=(denoise_opt == "On"), preprocess_prompt=(preprocess_opt == "On"), postprocess_output=(postprocess_opt == "On"))
        lang = language if (language and language != "Auto") else None
        kw = dict(text=text.strip(), language=lang, generation_config=gen_config)
        if mode == "clone":
            if not ref_audio: return None, "Please upload reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
        try:
            audio = model.generate(**kw)
            waveform = (audio[0] * 32767).astype(np.int16)

            # === SILENCE REMOVAL LOGIC ===
            if remove_silence == "On":
                float_wave = waveform.astype(np.float32) / 32767.0
                trimmed_wave, _ = librosa.effects.trim(float_wave, top_db=20)
                waveform = (trimmed_wave * 32767).astype(np.int16)
            # =============================

            os.makedirs("outputs", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("outputs", f"AS_STUDIO_{timestamp}.wav")
            sf.write(save_path, waveform, sampling_rate)
            return (sampling_rate, waveform), f"Done. Saved: {save_path}"
        except Exception as e: return None, f"Error: {e}"

    theme = gr.themes.Soft(primary_hue="slate", neutral_hue="slate")
    with gr.Blocks(theme=theme, title="🚀 AS STUDIO") as demo:
        gr.Markdown("# 🚀 AS STUDIO\n\nProfessional AI Voice Clone & Voice Design (600+ Languages)")
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                vc_text = gr.Textbox(label="Text to Synthesize", lines=4)
                vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                vc_lang = gr.Dropdown(label="Language", choices=_ALL_LANGUAGES, value="Auto")
                vc_dn = gr.Dropdown(label="Denoise", choices=["On", "Off"], value="On")
                vc_pp = gr.Dropdown(label="Preprocess", choices=["On", "Off"], value="On")
                vc_po = gr.Dropdown(label="Postprocess", choices=["On", "Off"], value="On")
                vc_silence = gr.Dropdown(label="Remove Silence Part", choices=["On", "Off"], value="Off")
                vc_btn = gr.Button("Generate", variant="primary")
                vc_audio = gr.Audio(label="Output")
                vc_status = gr.Textbox(label="Status")
                vc_btn.click(_gen_core, inputs=[vc_text, vc_lang, vc_ref_audio, gr.State(""), gr.State(32), gr.State(2.0), vc_dn, gr.State(1.0), gr.State(None), vc_pp, vc_po, vc_silence, gr.State("clone")], outputs=[vc_audio, vc_status])
    return demo

def main():
    parser = build_parser()
    args = parser.parse_args()
    model = OmniVoice.from_pretrained(args.model, device_map=args.device or get_best_device(), dtype=torch.float16, load_asr=True)
    demo = build_demo(model, args.model)
    demo.queue().launch(server_name=args.ip, server_port=args.port, share=True)
if __name__ == "__main__": main()
