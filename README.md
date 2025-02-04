# **DeepIntel AI** 🚀  
An advanced AI-driven research assistant powered by CrewAI, integrating Gemini and Groq for efficient information gathering, analysis, and content synthesis.  

## **Overview**  
**DeepIntel AI** is a multi-agent research system designed to:  
1. **Gather information** from various sources (**Researcher - Gemini**)  
2. **Analyze and process data** for insights (**Analyst - Groq**)  
3. **Synthesize refined content** for final output (**Writer - Gemini**)   

## **Tech Stack**  
- **CrewAI** – Multi-agent orchestration  
- **Gemini** – Large token window for research & synthesis  
- **Groq** – High-speed analysis with DeepSeek 

## **Installation**  
```bash  
pip install -r requirements.txt 
```

## **Usage**  
```python  
from crewai import Crew, Agent, Task  
from my_agents import researcher, analyst, writer  # Define your agents  

crew = Crew(agents=[researcher, analyst, writer])  
result = crew.kickoff()  
print(result)  
```

## **Why DeepIntel Agent?**  
✔ **Fast & Efficient** – Groq speeds up analysis.  
✔ **Deep Research** – Gemini processes large documents and ensures accuracy.  
✔ **Structured Output** – Ensures high-quality content generation.  

## **Future Enhancements**  
- Web search integration  
- Custom knowledge base  
- Fine-tuned LLMs for domain-specific research  
- Speeding up the whole execution process

