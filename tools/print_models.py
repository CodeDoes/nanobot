from nanobot.config.loader import load_config

config = load_config()
print('default model', config.agents.defaults.model)
print('models list', config.agents.defaults.models)
print('type', type(config.agents.defaults.models))
