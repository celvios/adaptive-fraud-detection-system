def create_html_slides():
    """Convert markdown to HTML slides"""
    with open('presentation_slides.md', 'r') as f:
        content = f.read()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Presentation</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .slide {{ page-break-after: always; margin-bottom: 50px; }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        </style>
    </head>
    <body>
        <div id="content">
    """
    
    # Split by slides and convert
    slides = content.split('---')[1:]  # Skip first empty part
    
    for slide in slides:
        html += f'<div class="slide">{slide.replace("###", "<h3>").replace("##", "<h2>").replace("**", "<strong>").replace("*", "<em>")}</div>'
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open('presentation_slides.html', 'w') as f:
        f.write(html)
    
    print("✅ Created presentation_slides.html")
    print("Open in browser and print to PDF or copy to PowerPoint")

if __name__ == "__main__":
    create_html_slides()