from replicate.prompts import system_content3, user_input3
from replicate.main import Replicator

replicator = Replicator(
    system=system_content3,
    task=user_input3,
)
response = replicator.run()
