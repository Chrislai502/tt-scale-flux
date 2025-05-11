import ImageReward as RM

# import utils.cache
# import utils.device

from base_verifier import BaseVerifier


class ImageRewardVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = [
        "overall_score",
    ]
    model_id = "ImageReward-v1.0"

    def __init__(self, **kwargs):
        self.verifier = RM.load(
            self.model_id,
            device="cuda:1",  # hard-code device.
            download_root="/pscratch/sd/c/chrislai/.cache/huggingface/",
        )
    
    def prepare_inputs(self, images, prompts):
        assert len(images) == len(prompts)

        # conversations = []
        # for prompt in prompts:
        #     conversations.append(self.prepare_conversations(prompt))

        # assert len(conversations) == len(images) == len(prompts)

        # prompts = [self.model.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
        # images = [[image] for image in images]
        inputs = {"images": images, "prompts": prompts}
        return inputs

    def score(self, inputs) -> list[dict[str, float]]:
        # TODO: might need to iterate `inputs` in batches depending on the resources.
        outputs = [{"overall_score": self.get_reward(prompt, image)} for prompt, image in zip(inputs["prompts"], inputs["images"])]
        return outputs

    def get_reward(self, prompt, image):
        return self.verifier.score(prompt, image)
