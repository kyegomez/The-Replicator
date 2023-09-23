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
            ai_role="A worker that creates other AI models",
        )

        prompt = f"{self.system} {self.task}"
        response = node.run(prompt)
        return response
