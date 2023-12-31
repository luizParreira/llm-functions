# LLM Functions

Create a rust library that uses huggingface's candle library to receive a users' prompt and generate a JSON that follows a specified schema. It is similar to OpenAI's functions, however in this library we actually enforce the result given by the LLM and ask it to correct if it gives us the wrong output.


# Architecture


GENERATOR - Responsible for instantiating the model and the functions to be used. And starting the chain of calls that will be used to generate the output.
ENFORCER - Responsible for enforcing the JSON output given by the model
