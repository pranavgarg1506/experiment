from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import yaml
from tools import tool_weather,ml_tools
from langchain.agents import Tool, initialize_agent, AgentType

load_dotenv()

print("All modules imported")

# function to load configuration from a YAML file 
def get_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


config = get_config()

tools = [
    Tool(name="Read Data", func=ml_tools.run_read_data, description="Read Dataset from a CSV file."),
    Tool(name="EDA", func=ml_tools.run_eda, description="Performs exploratory data analysis."),
    Tool(name="FeatureSelection", func=ml_tools.run_feature_selection, description="Selects important features."),
    Tool(name="Heatmap", func=ml_tools.run_heatmap, description="Creates a correlation heatmap."),
    Tool(name="WeatherFetcher",func=tool_weather.get_weather,description="Use this to get weather info by city name."),
]

# Create the agent
llm = ChatGoogleGenerativeAI(model=config['google_model_name'],google_api_key='')
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Use it
query = input("Ask something related to ML analysis: ")
print(agent.run(query))
