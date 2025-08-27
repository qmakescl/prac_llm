from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import asyncio
import time

# Ensure the GEMINI_API_KEY environment variable is set
from dotenv import load_dotenv
load_dotenv()


# Initialize the Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Create server parameters for Python execution
python_server_params = StdioServerParameters(
    command="deno",
    args=[
        "run",
        "-N",  # Using -N for network permissions
        "-R=node_modules",
        "-W=node_modules",
        "--node-modules-dir=auto",
        "--allow-scripts",  # Allow scripts for all packages
        "jsr:@pydantic/mcp-run-python",
        "stdio",
    ],
)

# Create server parameters for Airbnb
airbnb_server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@openbnb/mcp-server-airbnb",
        "--ignore-robots-txt",
    ],
)

async def run():
    # Connect to Python execution server
    async with stdio_client(python_server_params) as (python_read, python_write):
        async with ClientSession(python_read, python_write) as python_session:
            # Initialize the connection
            await python_session.initialize()
            
            # Get tools from MCP session and convert to Gemini Tool objects
            python_mcp_tools = await python_session.list_tools()
            
            # Clean up JSON schema to remove unsupported properties
            def clean_schema(schema):
                if isinstance(schema, dict):
                    schema_copy = schema.copy()
                    if 'additionalProperties' in schema_copy:
                        del schema_copy['additionalProperties']
                    if '$schema' in schema_copy:
                        del schema_copy['$schema']
                    
                    # Recursively clean nested properties
                    for key, value in schema_copy.items():
                        if isinstance(value, (dict, list)):
                            schema_copy[key] = clean_schema(value)
                    return schema_copy
                elif isinstance(schema, list):
                    return [clean_schema(item) for item in schema]
                else:
                    return schema
            
            python_tools = types.Tool(function_declarations=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": clean_schema(tool.inputSchema),
                }
                for tool in python_mcp_tools.tools
            ])
            
            # Connect to Airbnb server
            async with stdio_client(airbnb_server_params) as (airbnb_read, airbnb_write):
                async with ClientSession(airbnb_read, airbnb_write) as airbnb_session:
                    # Initialize the connection
                    await airbnb_session.initialize()
                    
                    # Get tools from MCP session and convert to Gemini Tool objects
                    airbnb_mcp_tools = await airbnb_session.list_tools()
                    airbnb_tools = types.Tool(function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": clean_schema(tool.inputSchema),
                        }
                        for tool in airbnb_mcp_tools.tools
                    ])
                    
                    # Combine tools from both servers
                    all_tools = types.Tool(function_declarations=[
                        *python_tools.function_declarations,
                        *airbnb_tools.function_declarations
                    ])
                    
                    # Create a prompt that includes both Python and Airbnb tasks
                    prompt = """
                    I need help with two tasks:
                    1. Calculate how many days are there between 2000-01-01 and 2025-03-18
                    2. Find apartments in Paris for 2 nights from 2025-03-28 to 2025-03-30
                    """
                    
                    # Send request with function declarations
                    response = client.models.generate_content(
                        model="gemini-2.5-pro-preview-03-25",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            tools=[all_tools],
                        ),
                    )
                    
                    # Process function calls without printing details
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call is not None:
                            function_call = part.function_call
                            if hasattr(function_call, 'name') and function_call.name:
                                # Execute the function call based on which server it belongs to
                                if function_call.name == "run_python_code":
                                    await python_session.call_tool(function_call.name, function_call.args)
                                elif function_call.name in [tool.name for tool in airbnb_mcp_tools.tools]:
                                    await airbnb_session.call_tool(function_call.name, function_call.args)
                    
                    # Print the final text response
                    print(response.text)