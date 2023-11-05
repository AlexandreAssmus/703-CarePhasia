
import vosk
import sys
import wave
import os
import getopt

try:
    opts,args = getopt.getopt(sys.argv[1:], "hi:d:o:",["ifile=","duration=","ofile="])
except: 
    print('vosk_recognizer.py -i <inputfile>')
    sys.exit(2)

for opt,arg in opts:
    if opt == '-h':
        print('vosk_recognizer.py -i <inputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        audio_file_path = arg
    elif opt in ("-d","--duration"):
        duration = int(arg)
    elif opt in ("-o","--ofile"):
        audio_output_file_path = arg



if ('duration' in locals()) or ('audio_output_file_path' in locals()) or len(sys.argv)==1:
    rec_command_line = "rec -c 1 -b 16 -r 16k tmp_input_file.wav"
    if('duration' in locals()):
        rec_command_line += ' trim 0 '+str(duration)
    os.system(rec_command_line)
    audio_file_path = "tmp_input_file.wav"


# Check if the provided file exists
if not os.path.isfile(audio_file_path):
    print(f"The file '{audio_file_path}' does not exist.")
    sys.exit(1)

# Open the audio file for reading
wf = wave.open(audio_file_path, "rb")

# Check audio sampling parameters
audio_params = wf.getparams()
if(audio_params.nchannels != 1) or (audio_params.sampwidth != 2) or (audio_params.framerate != 16000):
    print('converting audio format')
    wf.close()
    tmp_audio_file_path = 'tmp_'+audio_file_path
    audio_convert_command = 'sox '+audio_file_path+' -r 16000 -b 16 -c 1 '+tmp_audio_file_path
    os.system(audio_convert_command)
    wf = wave.open(tmp_audio_file_path, "rb")

# Initialize the Vosk recognizer with the downloaded model
model_path = "./vosk_models/vosk-model-en-us-0.22"
model = vosk.Model(model_path)

# Initialize the recognizer
rec = vosk.KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

# Read the audio file and recognize it
while True:
    data = wf.readframes(4000)  # Read 4,000 bytes at a time (adjust as needed)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        pass  # Adjust recognition output if needed

wf.close()
if 'tmp_audio_file_path' in locals():
    os.remove(tmp_audio_file_path)

result = rec.FinalResult()

# Print the transcription
print(result)