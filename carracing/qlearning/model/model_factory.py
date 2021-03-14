from qlearning.model.model_baseline import ModelBaseline
from qlearning.model.model_openai import ModelOpenAI


class ModelFactory(object):

    @classmethod
    def create_model(cls, architecture: str, input_size: int, input_frames: int, output_size: int):

        if architecture == "openai":
            return ModelOpenAI(input_size=input_size, input_frames=input_frames, output_size=output_size)

        elif architecture == "baseline":
            return ModelBaseline(input_size=input_size, input_frames=input_frames, output_size=output_size)
        
        else:
            raise Exception(f"Architecture {architecture} not supported")
