from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import yaml
from tools import tool_weather
from langchain.agents import Tool, initialize_agent, AgentType

load_dotenv()

print("All modules imported")

# function to load configuration from a YAML file 
def get_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


config = get_config()

# Define the tool
weather_tool = Tool(
    name="WeatherFetcher",
    func=tool_weather.get_weather,
    description="Use this to get weather info by city name."
)

# Create the agent
llm = ChatGoogleGenerativeAI(model=config['google_model_name'],google_api_key='')
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Use it
agent.invoke("What is the weather in Delhi?")