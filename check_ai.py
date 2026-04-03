import google.generativeai as genai

# আপনার API Key এখানে বসান
genai.configure(api_key="AIzaSyAy0CGzXf3UZ14FmmFTiLC29tKpzr0BD_U")

print("Checking available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")