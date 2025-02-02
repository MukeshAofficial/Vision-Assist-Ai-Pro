from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import asyncio
import base64
import json
import pyaudio
import io
import os
import sys
import traceback
import cv2
import PIL.Image
from websockets.asyncio.client import connect
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
import google.generativeai as genaii


load_dotenv()

app = FastAPI()

genaii.configure(api_key="API")


templates = Jinja2Templates(directory="templates")


class MessageRequest(BaseModel):
    message: str

@app.post("/get_response")
async def get_response(request: MessageRequest):
    user_message = request.message
    model = genaii.GenerativeModel('gemini-1.5-flash')
    rply = model.generate_content(f"{user_message} answer in one or 2 lines without any */ symbols for this medical related question ")
    return {"response": rply.text}

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

GOOGLE_API_KEY = "AIzaSyA-tAJWZDUcDpMEo8IfT3wEI9D39KMKVV8"
if not GOOGLE_API_KEY:
    raise ValueError("API_KEY environment variable not set. Check your .env file.")

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

MODEL = "models/gemini-2.0-flash-exp"

client = genai.Client(http_options={'api_version': 'v1alpha'}, api_key=GOOGLE_API_KEY)

INITIAL_PROMPT = """As a Visual Impairment Support Agent, your task is to assist visually impaired users by leveraging camera-based analysis and voice interaction. The system should be able to:

People Detection: Analyze the camera feed to detect how many people are present in the environment and provide a brief description of their appearance (e.g., "There are two people in front of you, one wearing a red shirt").
Object Recognition: Identify objects within the user’s surroundings and describe them (e.g., "You are holding a book titled '1984'").
Book Summary: When a book is detected, provide a short summary of the book and read the title and author aloud.
Text Reading: Use Optical Character Recognition (OCR) to read any visible text, such as on signs, documents, or labels, and announce it aloud.
Problem Solving: If the user requests assistance, analyze the context and provide solutions for basic tasks, such as reading math problems or identifying objects in the environment.
Voice Commands: Respond to voice instructions for navigation or interaction, ensuring all feedback is provided through spoken responses.
"""
CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()
        self.video_out_queue = asyncio.Queue()
        self.session = None
        self.conversation_history = []
        self._is_running = True # Added flag for stopping loops

    async def send_text(self):
        while self._is_running:  # Use flag to control loop
            text = await asyncio.to_thread(input, "You: ")
            if text.lower() == "q":
                break
            self.conversation_history.append({"role": "user", "parts": [{"text": text}]})
            await self.send_to_gemini()

    async def send_to_gemini(self):
        try:
            response = client.generate_text(
                model=MODEL,
                prompt=self.conversation_history,
                generation_config=CONFIG,
            )
            if response.result:
                print("Gemini Assist:", response.result)
                self.conversation_history.append({"role": "assistant", "parts": [{"text": response.result}]})
        except Exception as e:
            print(f"Error communicating with Gemini: {e}")

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        img = PIL.Image.fromarray(frame)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while self._is_running:  # Use flag to control loop
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            self.video_out_queue.put_nowait(frame)
        cap.release()

    async def send_frames(self):
        while self._is_running: # Use flag to control loop
            frame = await self.video_out_queue.get()
            await self.session.send(frame)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
            input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE
        )
        while self._is_running:  # Use flag to control loop
            data = await asyncio.to_thread(stream.read, CHUNK_SIZE)
            self.audio_out_queue.put_nowait(data)

    async def send_audio(self):
        while self._is_running: # Use flag to control loop
            chunk = await self.audio_out_queue.get()
            await self.session.send({"data": chunk, "mime_type": "audio/pcm"})

    async def receive_audio(self):
         while self._is_running: # Use flag to control loop
            try:
                async for response in self.session.receive():
                    server_content = response.server_content
                    if server_content and server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.text:
                                print(part.text, end="")
                                self.conversation_history.append({"role": "assistant", "parts": [{"text": part.text}]})
                            elif part.inline_data:
                                self.audio_in_queue.put_nowait(part.inline_data.data)
                    if server_content and server_content.turn_complete:
                        print("Turn complete")
                        while not self.audio_in_queue.empty():
                            self.audio_in_queue.get_nowait()
            except Exception as e:
                print(f"Error during audio recive: {e}")
                break

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True
        )
        while self._is_running: # Use flag to control loop
            try:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
            except Exception as e:
                 print(f"Error in playing audio {e}")
                 break

    async def run(self):
        self.conversation_history.append({"role": "user", "parts": [{"text": INITIAL_PROMPT}]})
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            self.session = session
            tasks = [
                self.send_text(),
                self.listen_audio(),
                self.send_audio(),
                self.get_frames(),
                self.send_frames(),
                self.receive_audio(),
                self.play_audio(),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
               print("Tasks were cancelled")
            finally:
                print("Closing audio loop and session")
                await self.close()


    async def close(self):
        self._is_running = False
        if self.session:
           await self.session.close()

class SimpleGeminiVoice:
    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.api_key = "AIzaSyA-tAJWZDUcDpMEo8IfT3wEI9D39KMKVV8"
        self.model = "gemini-2.0-flash-exp"
        self.uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.CHUNK = 512
        self.RATE = 16000
        self.is_running = True

    async def start(self):
        self.ws = await connect(
            self.uri, additional_headers={"Content-Type": "application/json"}
        )
        await self.ws.send(json.dumps({"setup": {"model": f"models/{self.model}"}}))
        await self.ws.recv(decode=False)
        print("Connected to Gemini, You can start talking now")

        tasks = [
            self.capture_audio(),
            self.stream_audio(),
            self.play_response()
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("tasks were cancelled")
        finally:
            print("Closing gemini voice and websocket")
            await self.close()



    async def capture_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        while self.is_running:
            data = await asyncio.to_thread(stream.read, self.CHUNK)
            await self.ws.send(
                json.dumps(
                    {
                        "realtime_input": {
                            "media_chunks": [
                                {
                                    "data": base64.b64encode(data).decode(),
                                    "mime_type": "audio/pcm",
                                }
                            ]
                        }
                    }
                )
            )

    async def stream_audio(self):
        while self.is_running:
            try:
                async for msg in self.ws:
                    response = json.loads(msg)
                    try:
                        audio_data = response["serverContent"]["modelTurn"]["parts"][0][
                            "inlineData"
                        ]["data"]
                        self.audio_queue.put_nowait(base64.b64decode(audio_data))
                    except KeyError:
                        pass
                    try:
                        turn_complete = response["serverContent"]["turnComplete"]
                    except KeyError:
                        pass
                    else:
                        if turn_complete:
                            print("\nEnd of turn")
                            while not self.audio_queue.empty():
                                self.audio_queue.get_nowait()
            except Exception as e:
                print(f"Error in stream audio : {e}")
                break

    async def play_response(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT, channels=self.CHANNELS, rate=24000, output=True
        )
        while self.is_running:
           try:
              data = await self.audio_queue.get()
              await asyncio.to_thread(stream.write, data)
           except Exception as e:
              print(f"Error in playing : {e}")
              break

    async def close(self):
        self.is_running = False
        if self.ws:
            await self.ws.close()
            


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/live", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})

@app.get("/video", response_class=HTMLResponse)
async def get_video(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})


active_voice_clients = {}
active_video_clients = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = SimpleGeminiVoice()
    task = asyncio.create_task(client.start())  #start voice assistant
    active_voice_clients[websocket] = (client,task)

    try:
        while True:
            message = await websocket.receive_text()
            print(f"Message from frontend: {message}")
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket in active_voice_clients:
            client,task = active_voice_clients.pop(websocket)
            task.cancel()  # Cancel the running tasks
            await client.close()  # Close the client resources
            print("closing voice assistant connection")
            


@app.websocket("/ws/video")
async def video_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_loop = AudioLoop()
    task = asyncio.create_task(audio_loop.run())
    active_video_clients[websocket] = (audio_loop,task)

    try:
        while True:
            message = await websocket.receive_text()
            print(f"Message from frontend: {message}")
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket in active_video_clients:
            audio_loop,task  = active_video_clients.pop(websocket)
            task.cancel() # Cancel the running tasks
            await audio_loop.close()  # close the video client
            print("Closing video analysis connection")