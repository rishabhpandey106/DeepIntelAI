import streamlit as st
import os
import litellm
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

litellm.set_verbose=True

# Initialize search tools
search_tool = SerperDevTool()
website_tool = WebsiteSearchTool(config=dict(
        embedder=dict(
            provider="google",  # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
    ))

# Function to get LLM based on selected model
def get_llm(model_choice):
    """Initialize LLM based on the selected model"""
    if model_choice == "Gemini":
        return LLM(
            model="gemini/gemini-2.0-flash-exp", 
            temperature=0.7,
        )
    elif model_choice == "Groq":
        # Replace with the actual initialization for Groq model
        return LLM(
            model="groq/deepseek-r1-distill-llama-70b",  # Replace with the correct model name for Groq
            temperature=0.6,
        )
    elif model_choice == "HuggingFace":
        return LLM(
            model="huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )
    else:
        raise ValueError("Unsupported model choice")

def create_agents(model_choice):
    """Create specialized research and analysis agents"""
    try:
        llm = get_llm(model_choice)
        
        researcher = Agent(
            role='Deep Research Specialist',
            goal='Conduct comprehensive research and gather detailed information',
            backstory="""Expert researcher skilled at discovering hard-to-find information 
            and connecting complex data points. Specializes in thorough, detailed research.""",
            tools=[search_tool, website_tool],
            llm=llm,
            verbose=True,
            max_iter=10,
            allow_delegation=False
        )
        
        analyst = Agent(
            role='Research Analyst',
            goal='Analyze and synthesize research findings',
            backstory="""Expert analyst skilled at processing complex information and 
            identifying key patterns and insights. Specializes in clear, actionable analysis.""",
            tools=[search_tool],
            llm=llm,
            verbose=True,
            max_iter=7,
            allow_delegation=False
        )
        
        writer = Agent(
            role='Content Synthesizer',
            goal='Create clear, structured reports from analysis',
            backstory="""Expert writer skilled at transforming complex analysis into 
            clear, engaging content while maintaining technical accuracy.""",
            llm=llm,
            verbose=True,
            max_iter=4,
            allow_delegation=False
        )
        
        return researcher, analyst, writer
    except Exception as e:
        st.error(f"Error creating agents: {str(e)}")
        return None, None, None

def create_tasks(researcher, analyst, writer, topic):
    """Create research tasks with clear objectives"""
    research_task = Task(
        description=f"""Research this topic thoroughly: {topic}
        
        Follow these steps:
        1. Find reliable sources and latest information
        2. Extract key details and evidence
        3. Verify information across sources
        4. Document findings with references""",
        agent=researcher,
        expected_output="Detailed research findings with sources"
    )
    
    analysis_task = Task(
        description=f"""Analyze the research findings about {topic}:
        
        Steps:
        1. Review and categorize findings
        2. Identify patterns and trends
        3. Evaluate source credibility
        4. Note key insights""",
        agent=analyst,
        context=[research_task],
        expected_output="Analysis of findings and insights"
    )
    
    synthesis_task = Task(
        description=f"""Create a clear report on {topic}:
        
        Include:
        - Executive Summary
        - Key Findings
        - Evidence
        - Conclusions
        - Specific questions asked by the user
        - Search volume, demand, search conversation
        - Top keywords
        - References""",
        agent=writer,
        context=[research_task, analysis_task],
        expected_output="Structured report with insights"
    )
    
    return [research_task, analysis_task, synthesis_task]

def run_research(topic, model_choice):
    """Execute the research process"""
    try:
        researcher, analyst, writer = create_agents(model_choice)
        if not all([researcher, analyst, writer]):
            raise Exception("Failed to create agents")
        
        tasks = create_tasks(researcher, analyst, writer, topic)
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        # Convert CrewOutput to string for consistency
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Deep Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

    # Sidebar for model choice
    st.sidebar.title("⚙️ Settings")
    model_choice = st.sidebar.radio(
        "Choose Model",
        ["Gemini", "Groq" , "HuggingFace"],
    )

    # Check if the API key for the selected model is set
    if model_choice == "Gemini" and not os.getenv("GEMINI_API_KEY"):
        st.sidebar.warning("⚠️ Gemini API key not found")
    elif model_choice == "Groq" and not os.getenv("GROQ_API_KEY"):
        st.sidebar.warning("⚠️ Groq API key not found")
    elif model_choice == "HuggingFace" and not os.getenv("HUGGINGFACE_API_KEY"):
        st.sidebar.warning("⚠️ HF API key not found")

    # Main content
    st.title("🔍 Deep Research Assistant")
    st.markdown(""" 
    This AI-powered research assistant conducts comprehensive research on any topic.
    It uses specialized agents to research, analyze, and synthesize information.
    """)
    
    # Input for research topic
    query = st.text_area(
        "Research Topic",
        placeholder="Enter your research topic (be specific)...",
        help="More specific queries yield better results"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        start_research = st.button("🚀 Start Research", type="primary")

    # Execute research
    if start_research and query:
        with st.spinner("🔍 Conducting research..."):
            result = run_research(query, model_choice)

        if isinstance(result, str) and result.startswith("Error:"):
            st.error(result)
        else:
            st.success("✅ Research Complete!")
            
            tab1, tab2 = st.tabs(["📊 Report", "ℹ️ About"])
            
            with tab1:
                st.markdown("### Research Report")
                st.markdown("---")
                st.markdown(str(result))  # Ensure result is converted to string
                
            with tab2:
                st.markdown(f"""
                ### Process:
                1. **Research**: Comprehensive source search
                2. **Analysis**: Pattern identification
                3. **Synthesis**: Report creation
                
                **Details:**
                - Model: {model_choice}
                - Tools: Web search, content analysis
                - Method: Multi-agent collaboration
                """)

    st.divider()
    st.markdown("*Built with CrewAI and Streamlit*")

if __name__ == "__main__":
    main()
