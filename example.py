from replicate.prompts import system_content3, user_input3
from replicate.main import Replicator


user = "Create a time series model for predicting the stock market, use the code interpreter tool"

replicator = Replicator(
    system=user_input3,
    task=user,
)
response = replicator.run()
