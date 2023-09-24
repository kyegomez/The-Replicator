from swarms import Worker

class Replicator:
    def __init__(
        self,
        system: str = None,
        task: str = None,
    ):
        self.system = system
        self.task = task

    def run(self):
        node = Worker(
            openai_api_key="",
            ai_name="The Replicator",
            ai_role="An AI that creates experimental AI models using pytorch for business and research purposes.",
        )

        prompt = f"{self.system} {self.task}"
        response = node.run(prompt)
        return response
