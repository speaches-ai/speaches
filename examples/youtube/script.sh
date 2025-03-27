#!/usr/bin/env bash

set -e

# NOTE: do not use any distil-* model other than the large ones as they don't work on long audio files for some reason.
export TRANSCRIPTION_MODEL=deepdml/faster-distil-whisper-large-v3.5 # or Systran/faster-whisper-tiny.en if you are running on a CPU for a faster inference.

# Ensure you have `speaches` running. If this is your first time running it expect to wait up-to a minute for the model to be downloaded and loaded into memory. You can run `curl localhost:8000/health` to check if the server is ready or watch the logs with `docker logs -f <container_id>`.
docker run --detach --gpus=all --publish 8000:8000 --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub ghcr.io/speaches-ai/speaches:latest-cuda
# or you can run it on a CPU
# docker run --detach --publish 8000:8000 --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub ghcr.io/speaches-ai/speaches:latest-cpu

# Download the audio from a YouTube video. In this example I'm downloading "The Evolution of the Operating System" by Asionometry YouTube channel. I highly checking this channel out, the guy produces very high content. If you don't have `youtube-dl`, you'll have to install it. https://github.com/ytdl-org/youtube-dl
youtube-dl --extract-audio --audio-format mp3 -o the-evolution-of-the-operating-system.mp3 'https://www.youtube.com/watch?v=1lG7lFLXBIs'

# Make a request to the API to transcribe the audio. The response will be streamed to the terminal and saved to a file. The video is 30 minutes long, so it might take a while to transcribe, especially if you are running this on a CPU. `Systran/faster-distil-whisper-large-v3` takes ~30 seconds on Nvidia L4. `Systran/faster-whisper-tiny.en` takes ~1 minute on Ryzen 7 7700X. The .txt file in the example was transcribed using `Systran/faster-distil-whisper-large-v3`.
curl -s http://localhost:8000/v1/audio/transcriptions -F "file=@the-evolution-of-the-operating-system.mp3" -F "model=$TRANSCRIPTION_MODEL" -F "language=en" -F "response_format=text" | tee the-evolution-of-the-operating-system.txt

# Here I'm using `aichat` which is a CLI LLM client. You could use any other client that supports attaching/uploading files. https://github.com/sigoden/aichat
aichat -m openai:gpt-4o -f the-evolution-of-the-operating-system.txt 'What companies are mentioned in the following Youtube video transcription? Responed with just a list of names'
# 1. OpenAI
# 2. General Motors Research Lab
# 3. IBM
# 4. Univac
# 5. MIT
# 6. Bell Labs
# 7. Honeywell
# 8. Intel
# 9. Digital Research
# 10. Apple
# 11. Microsoft
# 12. VisitCorp
# 13. Lotus
# 14. AT&T
# 15. Palm
# 16. Symbian
# 17. Nokia
# 18. Verizon
# 19. Singular
# 20. Google

aichat -m openai:gpt-4o -f the-evolution-of-the-operating-system.txt 'Provide a summary of key events and their dates from the following Youtube video transcription'
# Certainly! Here is a summary of key events and their dates from the video transcription:
#
# 1. **1956**: General Motors Research Lab developed batch computing software for the IBM 701 mainframe.
# 2. **1956**: Univac 1103a introduced the concept of the Interrupt.
# 3. **1959**: John McCarthy proposed the concept of time-sharing operating systems.
# 4. **1961**: MIT team led by Fernando Corvado developed a prototype time-sharing system on the IBM 709.
# 5. **1962**: MIT announced the Compatible Time-Sharing System (CTSS).
# 6. **1964**: MIT, Bell Labs, and General Electric began developing Multics.
# 7. **1964**: IBM announced the System 360 computer line.
# 8. **1969**: Bell Labs pulled out of the Multics project.
# 9. **1971**: Intel released the first microprocessor, the 4004.
# 10. **1973**: Intel released the updated 808 microprocessor.
# 11. **1974**: Intel released the 8080 microprocessor.
# 12. **1980**: IBM began a secret project to create the IBM PC.
# 13. **1981**: IBM PC was released with PC DOS, developed by Microsoft.
# 14. **1983**: Microsoft released the first version of Windows.
# 15. **1993**: Microsoft Office had 90% of the productivity market.
# 16. **1993**: Apple released the Newton PDA.
# 17. **1996**: Microsoft released Windows CE for PDAs.
# 18. **1998**: Major phone makers adopted the Symbian OS.
# 19. **2007**: Apple released the iPhone.
# 20. **2008**: Apple opened the App Store.
# 21. **2008**: Google pivoted Android to compete with iOS.
#
# These events highlight the evolution of operating systems from batch computing to time-sharing, the rise of personal computers, and the development of mobile operating systems.
