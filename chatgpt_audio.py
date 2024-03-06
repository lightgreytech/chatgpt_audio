#vosk for speech recognition (#pip install vosk)
#sounddevice for use of microphone on the pc (#pip install sounddevice)
#pyttsx3 for text to speech (#pip install py3-tts)
#openai for making use of the chatgpt engine (requires openai API key) - (#pip install openai)
#argparse (#pip install argparse)

import argparse
import queue
import sys
import json
import sounddevice as sd
import openai
from vosk import Model, KaldiRecognizer, SetLogLevel

import pyttsx3

SetLogLevel(-1)  # this disables annoying Vosk logs

def initialize_engine():
    engine = pyttsx3.init()
    # Set properties (optional)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 200)
    return engine

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

if __name__ == "__main__":
    # Initialize pyttsx3 engine
    engine = initialize_engine()

    q = queue.Queue()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="show list of audio devices and exit")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        "-f", "--filename", type=str, metavar="FILENAME",
        help="audio file to store recording to")
    parser.add_argument(
        "-d", "--device", type=int_or_str,
        help="input device (numeric ID or substring)")
    parser.add_argument(
        "-r", "--samplerate", type=int, help="sampling rate")
    parser.add_argument(
        "-m", "--model", type=str, help="language model; eg en-us, fr, nl; default is en-us")
    args = parser.parse_args(remaining)

    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, "input")
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info["default_samplerate"])

        if args.model is None:
            model = Model(lang="en-us")
        else:
            model = Model(lang=args.model)

        if args.filename:
            dump_fn = open(args.filename, "wb")
        else:
            dump_fn = None

        # Main line taking microphone input and converting speech to text...
        with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
                               dtype="int16", channels=1, callback=callback):
            print("#" * 80)
            print("Start chatting...")
            print("#" * 80)

            rec = KaldiRecognizer(model, args.samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result())["text"]
                    if text != "":
                        print(text)
                        openai.api_key_path = "apikey.txt"  # here, provide the path to your openAI key
                        response = openai.Completion.create(
                            model="text-davinci-003",  # this is the ChatGPT3 model
                            prompt=text,
                            temperature=0.7,
                            max_tokens=200,  # adjust this to control length of response; note the higher this number, the more credit is used up
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                        saytext = response["choices"][0]["text"]
                        print(saytext)
                        # use text to speech to communicate ChatGPT response
                        engine.say(saytext)
                        engine.runAndWait()
                if dump_fn is not None:
                    dump_fn.write(data)

    except KeyboardInterrupt:
        print("\nDone")
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ": " + str(e))