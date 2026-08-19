from agent.agent import NanoCodeAgent

agent = NanoCodeAgent()

response = agent.run("Which python version is used in this environment?")

print(response)