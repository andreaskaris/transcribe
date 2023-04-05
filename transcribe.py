import sounddevice as sd
import soundfile as sf
import whisper
import sys
import openai
from gtts import gTTS
import os

fs = 44100  # Sample rate
ch = 2
st = "PCM_24"
fn = "audio.wav"


openai.api_key = os.getenv('OPENAI_KEY')


def load_model():
    print("Loading model, this can take a while")
    model = whisper.load_model("medium")
    print("Model loaded")
    return model


def transcribe(model, fn):
    print("Transcribing, this can take a while")
    result = model.transcribe(fn, fp16=False, language='Spanish')
    print("Done transcribing")
    return result["text"]


def record(fn, fs, ch, st):
    with sf.SoundFile(fn, mode='w', samplerate=fs, channels=ch,
                      subtype=st) as file:
        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            file.write(indata.copy())
        with sd.InputStream(samplerate=fs, channels=ch,
                            callback=callback):
            input("Recording now - press return to end recording ")


conversation = [
    {"role": "system", "content": "You are a helpful language training " +
     "chatbot. Your answers are short and concise. You respect the user's " +
     "requested language and answer in the same language. When the user " +
     "makes mistakes such as bad grammar or bad vocabulary, you correct the " +
     " user."},
        ]


def send_message(message):
    global conversation
    conversation.append({"role": "user", "content": message})
    conversation.append({"role": "system", "content": "Correct any mistakes " +
                         "if there are, in the user's language of choice"})
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=conversation,
    )
    answer = completion.choices[0].message
    conversation.append(answer)
    return answer.content


def play(message):
    # The text that you want to convert to audio
    mytext = message
    # Language in which you want to convert

    language = 'es'

    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=mytext, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")

    # Playing the converted file
    os.system("play welcome.mp3")


def main():
    model = load_model()

    while True:
        i = input("Press 'r' to start recording, 'l' for log, "
                  "press 'q' to quit: ")
        if i == "r":
            record(fn, fs, ch, st)
            message = transcribe(model, fn)
            print("USER: {}".format(message))
            s = input("Do you want to send this message? (y/n) ")
            if s == "y":
                answer = send_message(message)
                print("CHATBOT: {}".format(answer))
                play(answer)
        elif i == "l":
            print(conversation)
        elif i == "q":
            sys.exit(0)


main()
