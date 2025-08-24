Hybrid AI Surveillance System: A Smart Co-Pilot for Security
============================================================

Welcome to the repository for our AI-powered surveillance system, built for the Honeywell Hackathon. This project is more than just a program; it's a proof-of-concept for a smarter, more efficient future in security monitoring. We've created a system that acts as an intelligent co-pilot, watching video feeds to detect both known and unknown threats in real-time.

The Challenge: Seeing Everything, All the Time
----------------------------------------------

The core problem we tackled is one of human limitation. A security guard watching a wall of monitors is faced with an impossible task: maintain perfect focus for hours on end, trying to spot a few critical seconds of activity within a sea of uneventful footage. It's exhausting, inefficient, and, most importantly, prone to error. The goal was to build a system that could automate this process, filtering out the noise to find the signals that truly matter.

Our Solution: A Hybrid AI with Two Brains
-----------------------------------------

Instead of relying on a single approach, we built a **hybrid AI system** that combines the strengths of two different models, creating a far more robust and comprehensive security net.

### 🧠 Model 1: The Rule-Based Guard (YOLOv5)

This is our precision expert. It's trained to find specific things we tell it to look for. Using the powerful **YOLOv5** object detector and a **SORT tracker**, it identifies and follows every person and bag in the frame. We then built custom logic on top of this to enforce specific rules:

*   **Loitering Detection**: Is a person lingering in a sensitive area for too long?
    
*   **Object Abandonment**: Has a bag been left behind with no one nearby?
    

### 🧠 Model 2: The Behavioral Analyst (Autoencoder)

This is our intuition expert. It's designed to catch the things we _didn't_ think to program. We trained this **Convolutional Autoencoder** on hours of completely normal video from the Avenue dataset. It doesn't know what a "person" or "bag" is, but it has become an expert on the normal visual "rhythm" of a scene. When anything disrupts that rhythm—sudden running, a crowd panicking, a car driving where it shouldn't—the model fails to reconstruct the scene properly. This failure, which we measure as a **reconstruction error**, becomes a powerful signal for a general, undefined anomaly.

The Journey: What I've Done, Step-by-Step
-----------------------------------------

Building this system was an exciting journey through the entire machine learning pipeline.

1.  **Phase 1: Foundation & Object Tracking**My first step was to get a basic object detector running. I set up the **YOLOv5** environment and successfully ran it on sample images and videos. I then integrated the **SORT tracking algorithm**, modifying the core detection script to assign a persistent ID to every person it saw. This was the crucial foundation for all subsequent logic.
    
2.  **Phase 2: Building the Rules**With tracking in place, I developed the logic for our rule-based alerts. I created a system to monitor the position and duration of each tracked ID, triggering a "loitering" alert if it stayed within a small area for too long. I then expanded this to detect "object abandonment" by analyzing the proximity between stationary objects (like backpacks) and people.
    
3.  **Phase 3: Training the Anomaly Expert**This was a parallel effort. I wrote a prepare\_data.py script to process the **Avenue Dataset**, extracting thousands of "normal" frames and converting them to a standard format. Then, I wrote a train\_autoencoder\_pytorch.py script to train our **Convolutional Autoencoder** on this data. After several epochs, I had a trained anomaly\_detector.pth model ready for integration.
    
4.  **Phase 4: Integration & The Final Dashboard**This is where everything came together. I created the final dashboard.py script, a **Streamlit application** that loads both the YOLOv5 model and our new PyTorch autoencoder. I engineered the main processing loop to run both models in parallel on every frame of the video. The final step was building the user interface, adding sliders, checkboxes, a live anomaly score chart, and a unified alert log to create the powerful, interactive dashboard you see today.
    

Setup and Usage Instructions
----------------------------

Follow these steps to get the system running on your own machine.

### 1\. Prerequisites

*   Python 3.8+
    
*   A virtual environment is highly recommended.
    

### 2\. Installation

First, clone the necessary yolov5 repository into your main project folder.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone https://github.com/ultralytics/yolov5   `

Then, install all the required Python libraries using the requirements.txt file.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 3\. Data Preparation & Model Training

Before you can run the dashboard, you need to prepare the data and train the anomaly model._(Note: A pre-trained anomaly\_detector.pth is included, so you can skip these steps if you just want to run the demo.)_

**Download the Avenue Dataset** and place the training videos in the avenue\_dataset/training\_videos/ folder.

**Run the preparation script:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python prepare_data.py   `

**Run the training script:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python train_autoencoder_pytorch.py   `

### 4\. Launch the Dashboard

Now, you're ready to start the surveillance system.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run dashboard.py   `

A new tab should open in your web browser with the application. Use the sidebar to configure your settings and click **"Start Analysis"**.

Technologies Used
-----------------

*   **Python**
    
*   **PyTorch** (for both YOLOv5 and the Autoencoder)
    
*   **OpenCV** (for all video and image processing)
    
*   **Streamlit** (for the interactive web dashboard)
    
*   **SORT** (for real-time object tracking)
    
*   **NumPy**, **Pillow**, **Matplotlib**