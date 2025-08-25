import subprocess
import sys

def convert_to_pptx():
    """Convert markdown slides to PowerPoint"""
    try:
        # Install pandoc if needed
        subprocess.run([sys.executable, "-m", "pip", "install", "pypandoc"], check=True)
        
        # Convert using pandoc
        subprocess.run([
            "pandoc", 
            "presentation_slides.md", 
            "-o", "presentation_slides.pptx",
            "-t", "pptx"
        ], check=True)
        
        print("✅ Converted to presentation_slides.pptx")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Manual conversion needed - copy content to PowerPoint")

if __name__ == "__main__":
    convert_to_pptx()