This project uses agentic AI systems and generative modeling to recreate historically accurate portraits and short biographies of influential figures throughout history. Each generated output includes a unique, AI-generated illustration and a concise description summarizing the individual’s contributions and historical impact.

⚙️ How It Works

Developed multi-agent scripts that coordinate between text generation and image synthesis models.

Used prompt engineering to produce contextually rich and visually accurate character depictions.

The system combines LLM-based historical summarization with text-to-image generation to create cohesive outputs.

🧩 Project Structure
historical-figures-AI/
│
├── multi_agent.py               # Coordinates agents for text & image generation  
├── multi_agent_prompts.py       # Custom prompt logic for LLM and image synthesis  
├── tools/                       # Utility scripts for model setup and output management  
├── textbook/                    # Reference dataset containing historical text samples  
├── output/                      # Generated portraits and biographies  
└── README.md                    # You're here!

📖 Example Output

Figure: Christopher Columbus
Description:
Christopher Columbus, an Italian explorer, is credited with the discovery of the New World in 1492. He convinced the Spanish monarchs Ferdinand II and Isabella I to fund his westward expedition to Asia. Despite misconceptions about Earth’s size, Columbus’s voyages opened the door to European exploration and colonization of the Americas.

🧠 Tech Stack

Python

Generative AI (image + text models)

Prompt Engineering

Multi-Agent System Architecture

PyTorch / FastAI (for model coordination)

🚀 Future Plans

Add interactive UI for generating new figures on-demand.

Expand dataset to include lesser-known historical figures and modern contributors.

Integrate text-to-speech and timeline visualization for educational use.
