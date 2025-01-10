import google.generativeai as genai
import os
import telebot
from collections import defaultdict
from datetime import datetime, timedelta
from boto3 import client as boto3_client
from pydub import AudioSegment
import keyring
import gc
from contextlib import contextmanager
import logging
import requests
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def validate_google_api_key(api_key):
#     """Validate Google API key by attempting to make an actual API call"""
#     try:
#         genai.configure(api_key=api_key)
#         # Create a test model
#         test_model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
#         # Actually test the API by making a simple request
#         test_response = test_model.generate_content("Test.")
#         test_response.text  # This will fail if the API key is invalid
#         logger.info('Gemini API key fully validated')
#         return True
#     except Exception as e:
#         logger.error(f"Google API key validation failed: {str(e)}")
#         return False

def get_deployment_mode():
    """Get deployment mode from AWS Parameter Store or default to local"""
    try:
        # First check if we're actually in AWS by trying to access instance metadata
        requests.get('http://169.254.169.254/latest/meta-data/', timeout=1)
        logger.info("Running on AWS instance")
        
        # If we are in AWS, use instance role credentials
        return 'aws'
        
    except (requests.RequestException, Exception) as e:
        logger.info("Not running on AWS instance, defaulting to local mode")
        return 'local'

DEPLOYMENT_MODE = get_deployment_mode()

def get_aws_parameter(parameter_name):
    """Retrieve parameter from AWS Parameter Store"""
    try:
        ssm = boto3_client('ssm', region_name='us-east-1')
        response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return response['Parameter']['Value']
    except Exception as e:
        logger.info(f"Could not retrieve parameter from AWS: {e}")
        return None

# Get credentials based on deployment mode
deployment_mode = get_deployment_mode()
if deployment_mode == 'aws':
    logger.info("Running in AWS mode - using instance role credentials")
    # Use instance role credentials for AWS services
    telegram_token = get_aws_parameter('galebach_spanish_bot_token')
    genai.configure(api_key=get_aws_parameter('GOOGLE_AI_API_KEY'))
    
    # Create AWS clients using instance role credentials
    polly = boto3_client('polly', region_name='us-east-1')

else:
    logger.info("Running in local mode - getting credentials from keyring")
    telegram_token = keyring.get_password('api_telegram_spanish_tutor', 'galebach_spanish_bot_token')
    genai.configure(api_key=keyring.get_password('api_default_GOOGLE_AI_API_KEY', 'GOOGLE_API_KEY'))

    # Only set these variables in local mode
    aws_access_key = keyring.get_password('api_default_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID')
    aws_secret_key = keyring.get_password('api_default_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY')
    aws_region = keyring.get_password('api_default_AWS_DEFAULT_REGION', 'AWS_DEFAULT_REGION')

    # Configure Polly with explicit credentials for local mode
    polly = boto3_client('polly',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

# Verify all credentials were retrieved
required_credentials = {
    'Telegram Token': telegram_token,
}

for cred_name, cred_value in required_credentials.items():
    if not cred_value:
        logger.error(f"Failed to retrieve {cred_name}")
        raise ValueError(f"Missing required credential: {cred_name}")

# Configure services with credentials
bot = telebot.TeleBot(telegram_token)

@contextmanager
def model_context():
    """Context manager for Gemini model cleanup"""
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        yield model
    finally:
        del model
        gc.collect()

# Initialize Gemini model
model = genai.GenerativeModel('models/gemini-2.0-flash-exp')

GEMINI_PROMPT = """Your role is to help them practice and learn Spanish. 
Listen to their audio messages and respond naturally to what they say. ALWAYS respond in SPANISH!!!
Keep in mind that they will make mistakes. Avoid starting with generic greetings or phrases like "¡Hola!", "¡Qué interesante!", "¡Excelente pregunta!", etc. 

Provide detailed, informative responses that:
- Include at least 3-4 complete sentences
- Explain concepts thoroughly
- Give examples when relevant
- Include cultural context when appropriate
- Never exceed 2000 characters
- Always end with an engaging follow-up question

Instead of just answering directly, expand on the topic and provide rich context."""

# Add conversation history storage
class ConversationManager:
    def __init__(self, expiry_minutes=30):
        self.conversations = defaultdict(list)
        self.expiry_minutes = expiry_minutes
        
    def add_message(self, user_id, role, content):
        # Clean expired conversations first
        self._clean_expired()
        
        self.conversations[user_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
    
    def get_history(self, user_id):
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in self.conversations[user_id]
        ]
    
    def _clean_expired(self):
        current_time = datetime.now()
        for user_id in list(self.conversations.keys()):
            self.conversations[user_id] = [
                msg for msg in self.conversations[user_id]
                if current_time - msg['timestamp'] < timedelta(minutes=self.expiry_minutes)
            ]
    
    def clear_history(self, user_id):
        """
        Clear conversation history for a specific user
        """
        if user_id in self.conversations:
            del self.conversations[user_id]

# Initialize conversation manager after bot initialization
conversation_manager = ConversationManager()

# Extract common response generation logic into a helper function
def generate_gemini_response(prompt, user_id, input_content, file=None):
    """Generate response from Gemini model with conversation history"""
    try:
        with model_context() as current_model:
            # Get conversation history
            history = conversation_manager.get_history(user_id)
            
            # Create the prompt with history context
            full_prompt = prompt + "\n\nConversation history:\n"
            for msg in history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                full_prompt += f"{role}: {msg['content']}\n"
            
            # Generate response
            if file:
                response = current_model.generate_content([full_prompt, file])
            else:
                response = current_model.generate_content(full_prompt + f"\nUser: {input_content}")
            
            # Store interaction history
            conversation_manager.add_message(user_id, 'user', input_content)
            conversation_manager.add_message(user_id, 'assistant', response.text)
            
            return response.text
    finally:
        gc.collect()

def synthesize_speech(text):
    """
    Generate speech from text using Amazon Polly
    Returns: Path to temporary WAV file
    """
    try:
        # Use neural engine with Spanish voice
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='pcm',
            VoiceId='Lupe',  # Mexican Spanish female voice
            Engine='neural',
            LanguageCode='es-MX',
            SampleRate='16000'
        )
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Add debug logging
            print(f"Creating temporary file: {temp_file.name}")
            
            # Read audio stream data
            audio_data = response['AudioStream'].read()
            print(f"Received audio data length: {len(audio_data)} bytes")
            
            try:
                # Convert PCM to WAV
                audio = AudioSegment(
                    data=audio_data,
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                audio.export(temp_file.name, format="wav")
                print(f"Successfully exported WAV file to: {temp_file.name}")
                return temp_file.name
            except Exception as e:
                print(f"Error in audio conversion: {str(e)}")
                raise
                
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")
        raise

@bot.message_handler(content_types=['text'])
def handle_text(message):
    try:
        # Check if user wants to clear history
        if message.text.lower() == "clear history":
            conversation_manager.clear_history(message.from_user.id)
            bot.reply_to(message, "¡Historial de conversación borrado! Empecemos de nuevo.")
            return
            
        # Get text response from Gemini
        response = generate_gemini_response(
            GEMINI_PROMPT, 
            message.from_user.id,
            message.text
        )
        
        # Generate audio response
        audio_file = synthesize_speech(response)
        
        # Send text and audio responses
        bot.reply_to(message, response)
        with open(audio_file, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)
            
        # Cleanup temporary file
        os.unlink(audio_file)
            
    except Exception as e:
        bot.reply_to(message, f"Sorry, there was an error processing your message: {str(e)}")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_audio(message):
    try:
        # Get file info
        if message.voice:
            file_info = bot.get_file(message.voice.file_id)
            mime_type = 'audio/ogg'  # Telegram voice messages are typically OGG
        else:
            file_info = bot.get_file(message.audio.file_id)
            mime_type = 'audio/mpeg'  # Default for audio files

        # Create temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
            downloaded_file = bot.download_file(file_info.file_path)
            temp_file.write(downloaded_file)
            temp_file_path = temp_file.name

        # Upload file to Gemini with mime_type specified
        gemini_file = genai.upload_file(path=temp_file_path, mime_type=mime_type)
        
        # Generate response using the helper function
        response = generate_gemini_response(
            GEMINI_PROMPT,
            message.from_user.id,
            'Audio message sent',
            gemini_file
        )
        
        # Generate audio response
        audio_file = synthesize_speech(response)
        
        # Send text and audio responses
        bot.reply_to(message, response)
        with open(audio_file, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)
            
        # Cleanup temporary files
        os.unlink(temp_file_path)
        os.unlink(audio_file)

    except Exception as e:
        bot.reply_to(message, f"Sorry, there was an error processing your audio: {str(e)}")

# Start the bot
if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling()
